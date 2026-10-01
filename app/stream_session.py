"""Transport-independent state machine for one `/asr_stream_api_v1` stream.

Everything here is synchronous and meant to run in a worker thread: the
caller hands in whatever audio has arrived and gets back the messages to
send. Model calls go through the shared InferGate one step at a time, so
several streams (and batch requests) interleave instead of blocking the
event loop or calling the engine concurrently.
"""

import logging
import time
from typing import List, Optional

import numpy as np

from infer_gate import InferGate
from script_convert import DEFAULT_OUTPUT_SCRIPT, StreamingConverter, convert_text
from textproc import (
    clean_transcript,
    detect_and_fix_repetitions,
    detect_hallucination,
    is_chinese_token,
    keep_streamed_ending,
    split_text_to_tokens,
)
from vad import StreamVad

log = logging.getLogger("r2t2-service")

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 2560  # 160 ms
FIRST_CHUNK_SAMPLES = 2 * CHUNK_SAMPLES  # 160 ms step + 160 ms lookahead
# 推論跟不上送音節奏時，一步最多併 8 個 chunk（1.28 秒）追進度。
# 每一步的運算量幾乎不隨 chunk 長度增加，併得越多越能在多路同時講話時跟上。
MAX_CATCHUP_CHUNKS = 8
# 開口前保留的音訊。VAD 要累積約 100ms 語音才會判定開口，少了這段會吃掉句首。
PREROLL_SAMPLES = 3 * CHUNK_SAMPLES
# 模型大約每 80ms 音訊出一個 token。
SAMPLES_PER_TOKEN = 1280
RECENT_TOKENS = 10
UNFIXED_TOKEN_NUM = 1
# 一句超過這個長度就不做句末重辨識，直接用串流結果。
FINAL_PASS_MAX_SAMPLES = 30 * SAMPLE_RATE


def unfixed_tail(state) -> str:
    """Text the last step decoded but had not yet committed as fixed."""
    if state is None:
        return ""
    fixed = "".join(state.chunk_text).strip()
    latest = state.text.split("|")[0].strip()
    return latest[len(fixed):] if latest.startswith(fixed) else ""


class StreamSession:
    """One client stream: buffering, VAD segmentation, incremental decode.

    Each sentence ends with a ``reset: true`` message. Besides the last
    delta it carries ``final_text``: the whole sentence re-decoded in one
    pass. Streaming commits text before it has heard the rest of the
    sentence and cannot take it back, so the one-pass result is the more
    accurate of the two and is what a client should keep.
    """

    def __init__(
        self,
        model,
        gate: InferGate,
        *,
        request_id: str,
        context: str = "",
        language: Optional[str] = None,
        vad: Optional[StreamVad] = None,
        final_pass: bool = True,
        output_script: str = DEFAULT_OUTPUT_SCRIPT,
    ) -> None:
        self._model = model
        self._gate = gate
        self._request_id = request_id
        self._context = context
        self._language = language
        self._vad = vad
        self._final_pass = final_pass
        self._output_script = output_script
        self._converter = StreamingConverter(output_script)

        self._buf = np.zeros((0,), dtype=np.float32)
        self._preroll = np.zeros((0,), dtype=np.float32)
        # 有 VAD 時只在偵測到語音後才解碼；沒有 VAD 就從頭一路解碼。
        self._in_speech = vad is None
        self._state = None
        self._seg_audio: List[np.ndarray] = []
        self._seg_samples = 0
        self._sent_text = ""
        self._recent_tokens: List[str] = []
        self._budget_boost = 0.0

    # ------------------------------------------------------------------ API

    def feed(self, pcm: np.ndarray) -> List[dict]:
        """Consume newly arrived audio; return the messages to send.

        VAD always runs on every 160 ms chunk. Only decoding is batched: when
        a call brings several chunks (inference fell behind), consecutive
        speech chunks are decoded in one step, but never across a sentence
        boundary. Batching before the VAD would let a sentence end and the
        next one start inside one step, and the second sentence would be lost.
        """
        if pcm.shape[0]:
            self._buf = np.concatenate([self._buf, pcm])
        msgs: List[dict] = []
        pending: List[np.ndarray] = []
        pending_samples = 0
        vad_ms = 0.0

        def flush(ended: bool = False) -> None:
            nonlocal pending, pending_samples, vad_ms
            if not pending:
                return
            chunk = np.concatenate(pending)
            pending, pending_samples = [], 0
            msgs.append(self._wrap(self._localize(self._decode_step(chunk, ended, vad_ms))))
            vad_ms = 0.0

        while True:
            n = self._next_chunk_size(bool(pending))
            if n == 0:
                break
            chunk = self._buf[:n]
            self._buf = self._buf[n:]

            ended = False
            if self._vad is not None:
                t0 = time.monotonic()
                started, ended = self._vad.detect(chunk)
                vad_ms += (time.monotonic() - t0) * 1000
                if not self._in_speech:
                    if not started:
                        self._preroll = np.concatenate([self._preroll, chunk])[-PREROLL_SAMPLES:]
                        msgs.append(
                            self._wrap(
                                {
                                    "text": "",
                                    "asr_cost_ms": 0.0,
                                    "reset": False,
                                    "total_cost_ms": round(vad_ms, 1),
                                }
                            )
                        )
                        vad_ms = 0.0
                        continue
                    self._in_speech = True
                    pending.append(self._preroll)
                    pending_samples += self._preroll.shape[0]
                    self._preroll = np.zeros((0,), dtype=np.float32)

            pending.append(chunk)
            pending_samples += chunk.shape[0]
            if ended or pending_samples >= MAX_CATCHUP_CHUNKS * CHUNK_SAMPLES:
                flush(ended)
        flush()
        return msgs

    def finish(self) -> dict:
        """Client signalled end of stream: flush and close the last sentence."""
        tail = self._buf
        self._buf = np.zeros((0,), dtype=np.float32)
        if not self._in_speech or (self._state is None and tail.shape[0] == 0):
            return self._wrap({"text": "", "reset": True, "final_text": ""})
        first_delta = self._decode(tail) if tail.shape[0] else ""
        return self._wrap(
            self._localize(self._close_segment(still_speaking=False, first_delta=first_delta))
        )

    # ------------------------------------------------------------ internals

    def _localize(self, msg: dict) -> dict:
        """Turn a message's text into the script the client asked for."""
        msg["text"] = self._converter.push(msg["text"])
        if msg.get("reset"):
            msg["final_text"] = convert_text(msg["final_text"], self._output_script)
            self._converter.reset()
        return msg

    def _wrap(self, msg: dict) -> dict:
        return {"status": "success", "requestId": f"{self._request_id}", "msg": msg}

    def _next_chunk_size(self, decoding_started: bool) -> int:
        avail = self._buf.shape[0]
        # 沒有 VAD 時沿用原協議：第一步要湊滿 160ms + 160ms lookahead 才開始解碼。
        first = self._vad is None and self._state is None and not decoding_started
        need = FIRST_CHUNK_SAMPLES if first else CHUNK_SAMPLES
        return need if avail >= need else 0

    def _decode_step(self, chunk: np.ndarray, ended: bool, vad_ms: float) -> dict:
        t0 = time.monotonic()
        delta = self._decode(chunk)
        msg = {"text": delta, "reset": False}

        halluc, reason = detect_hallucination(self._sent_text)
        if ended:
            msg = self._close_segment(still_speaking=False, first_delta=delta)
        elif halluc:
            log.info("requestId=%s: hallucination reset, reason=%s", self._request_id, reason)
            msg = self._close_segment(still_speaking=True, first_delta=delta, hallucinated=True)
        msg["asr_cost_ms"] = round((time.monotonic() - t0) * 1000, 1)
        msg["total_cost_ms"] = round(vad_ms + msg["asr_cost_ms"], 1)
        return msg

    def _new_state(self):
        return self._model.init_streaming_state(
            context=self._context,
            language=self._language,
            unfixed_chunk_num=0,
            unfixed_token_num=UNFIXED_TOKEN_NUM,
            chunk_size_sec=CHUNK_SAMPLES / SAMPLE_RATE,
        )

    def _token_budget(self, samples: int) -> int:
        """How many tokens one step may emit for ``samples`` of new audio."""
        base = max(1, samples // SAMPLES_PER_TOKEN)
        cap = min(32, max(4, 2 * base))
        budget = base + self._budget_boost
        # 中文一個字常拆成多個 token，預算加倍才不會越講越落後。
        if self._recent_tokens and is_chinese_token(self._recent_tokens[-1]):
            budget = 2 * base
        return int(min(cap, budget))

    def _decode(self, chunk: np.ndarray) -> str:
        """Run one incremental decode step; return the newly fixed text."""
        self._seg_audio.append(chunk)
        self._seg_samples += chunk.shape[0]
        budget = self._token_budget(chunk.shape[0])

        # 建立 state 會用到 tokenizer，它不能跨執行緒同時使用，所以也放在鎖內。
        with self._gate.run():
            if self._state is None:
                self._state = self._new_state()
            state = self._state
            if chunk.shape[0] >= CHUNK_SAMPLES:
                state.chunk_size_samples = chunk.shape[0]
                _, fixed = self._model.streaming_transcribe_no_reset(chunk, state, budget, False)
            else:
                # 結束時不足一個 chunk 的尾巴
                state.buffer = np.concatenate([state.buffer, chunk])
                fixed = self._model.finish_streaming_transcribe_no_reset(state, budget)
        fixed = fixed.split("|")[0]

        if len(fixed) <= len(self._sent_text):
            if not (self._recent_tokens and is_chinese_token(self._recent_tokens[-1])):
                self._budget_boost += 0.5
            return ""
        delta = fixed[len(self._sent_text):]
        self._sent_text = fixed
        self._budget_boost = 0.0
        self._recent_tokens = (self._recent_tokens + split_text_to_tokens(delta))[-RECENT_TOKENS:]
        return delta

    def _close_segment(
        self,
        *,
        still_speaking: bool,
        first_delta: str = "",
        hallucinated: bool = False,
    ) -> dict:
        """End the current sentence and start the next one from a clean state."""
        tail = "" if hallucinated else unfixed_tail(self._state)
        streamed = detect_and_fix_repetitions(self._sent_text + tail)
        final_text = streamed
        did_final_pass = False
        if (
            self._final_pass
            and not hallucinated
            and 0 < self._seg_samples <= FINAL_PASS_MAX_SAMPLES
        ):
            audio = np.concatenate(self._seg_audio)
            try:
                with self._gate.run():
                    result = self._model.transcribe(
                        audio=[(audio, SAMPLE_RATE)],
                        context=self._context,
                        language=[self._language] if self._language else None,
                        return_time_stamps=False,
                    )
                final_text = keep_streamed_ending(clean_transcript(result[0].text), streamed)
                did_final_pass = True
            except Exception:
                log.exception("requestId=%s: final pass failed, keeping streamed text", self._request_id)

        self._state = None
        self._seg_audio = []
        self._seg_samples = 0
        self._sent_text = ""
        self._recent_tokens = []
        self._budget_boost = 0.0
        self._in_speech = still_speaking or self._vad is None
        return {
            "text": first_delta + tail,
            "reset": True,
            "final_text": final_text,
            "final_pass": did_final_pass,
        }
