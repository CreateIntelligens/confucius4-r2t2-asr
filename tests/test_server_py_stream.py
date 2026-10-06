"""Run server.py's /asr_stream_api_v1 handler against a fake model.

server.py is what the main deployment (.35, vLLM) runs, and vLLM cannot be
installed on the dev machines, so the streaming loop is exercised here with
a model that only costs time. Time is scaled down: one 160 ms chunk is sent
every SEND_INTERVAL and every decode step costs STEP_COST, so three streams
need more than real time if each chunk is decoded on its own, the way .35
fell behind (2026-10-05: three streams, 330 ms per step, sentence ends
arriving 2.5 s, then 4 s, then 16 s late and the last sentences lost).
"""

import asyncio
import importlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("sanic")

ROOT = Path(__file__).resolve().parents[1]
SEND_INTERVAL = 0.04
STEP_COST = 0.03
CHUNK_BYTES = 5120
SENTENCE_CHUNKS = 12
SILENCE_CHUNKS = 4
SENTENCES = 3


@pytest.fixture(scope="module")
def server():
    # server.py parses the command line when imported
    argv, sys.argv = sys.argv, ["server.py"]
    sys.path.insert(0, str(ROOT))
    try:
        module = importlib.import_module("server")
    finally:
        sys.path.remove(str(ROOT))
        sys.argv = argv
    return module


class SlowModel:
    """Every decode step costs STEP_COST no matter how much audio it covers."""

    def __init__(self):
        self.steps = 0

    def init_streaming_state(self, **kwargs):
        return SimpleNamespace(
            buffer=np.zeros((0,), dtype=np.float32),
            chunk_size_sec=0.16,
            chunk_size_samples=2560,
            text="",
            n=0,
        )

    def streaming_transcribe_no_reset(self, pcm, state, max_new_tokens, rollback=False):
        state.buffer = np.concatenate([state.buffer, pcm])
        fixed = ""
        while state.buffer.shape[0] >= state.chunk_size_samples:
            state.buffer = state.buffer[state.chunk_size_samples:]
            time.sleep(STEP_COST)
            self.steps += 1
            state.n += 1
            # distinct characters, so server.py's repetition check never fires
            state.text = "".join(chr(0x4E00 + i) for i in range(state.n))
            fixed = state.text[:-1]
        return state.text, fixed

    def finish_streaming_transcribe_no_reset(self, state, max_new_tokens):
        time.sleep(STEP_COST)
        return state.text


class ChunkVad:
    """Speech ends on the last chunk of every sentence (counted in 160 ms units)."""

    def __init__(self):
        self.samples = 0

    def detect_chunk(self, pcm):
        before, self.samples = self.samples, self.samples + len(pcm)
        period = (SENTENCE_CHUNKS + SILENCE_CHUNKS) * 2560
        ends = range(SENTENCE_CHUNKS * 2560, self.samples + 1, period)
        return [SimpleNamespace(is_speech_end=True) for end in ends if end > before]


class VadTemplate(ChunkVad):
    """server.py clones the shared VAD per connection with copy.copy + reset()."""

    audio_feat = None
    postprocessor = None

    def reset(self):
        self.samples = 0


class FakeWs:
    """Client side: sends a header, audio at a fixed pace, then EOS."""

    def __init__(self, request_id):
        self.inbox = asyncio.Queue()
        self.sent = []
        self.closed = False
        self.request_id = request_id
        self.speech_ends = []

    async def feed(self):
        await self.inbox.put(json.dumps({"requestId": self.request_id, "secret_key": "test0102", "language": "zh"}))
        start = time.monotonic()
        n = SENTENCES * (SENTENCE_CHUNKS + SILENCE_CHUNKS)
        for i in range(n):
            await self.inbox.put(np.zeros(CHUNK_BYTES // 2, dtype=np.int16).tobytes())
            if i % (SENTENCE_CHUNKS + SILENCE_CHUNKS) == SENTENCE_CHUNKS - 1:
                self.speech_ends.append(time.monotonic())
            await asyncio.sleep(max(0, start + (i + 1) * SEND_INTERVAL - time.monotonic()))
        self.eos_at = time.monotonic()
        await self.inbox.put("YOUDAO_ONETIME_ASR_STREAM_EOS")

    async def recv(self):
        return await self.inbox.get()

    async def send(self, data):
        self.sent.append((time.monotonic(), json.loads(data)))

    async def close(self, code=1000, reason=""):
        self.closed = True


def run_streams(server, n):
    async def main():
        server.asr_model = SlowModel()
        server.stream_vad = VadTemplate()
        server.inference_scheduler.start()
        try:
            clients = [FakeWs(f"r{i}") for i in range(n)]
            await asyncio.gather(
                *(server.asr_stream_api_v1(None, ws) for ws in clients),
                *(ws.feed() for ws in clients),
            )
            return clients
        finally:
            server.inference_scheduler.stop()

    return asyncio.run(main())


def resets(ws):
    return [(t, m["msg"]) for t, m in ws.sent if isinstance(m.get("msg"), dict) and m["msg"].get("reset")]


def test_three_streams_keep_up_with_real_time(server):
    for ws in run_streams(server, 3):
        ends = resets(ws)
        # every sentence plus the EOS reset arrives
        assert len(ends) == SENTENCES + 1, ws.request_id
        # a sentence end reaches the client soon after the speaker stops,
        # instead of drifting further behind with every sentence
        lags = [t - spoken for (t, _), spoken in zip(ends, ws.speech_ends)]
        assert max(lags) < 6 * SEND_INTERVAL, lags
        assert ends[-1][0] - ws.eos_at < 6 * SEND_INTERVAL


def test_merged_steps_still_stream_every_character(server):
    """Merging a backlog into one step must not drop or repeat text."""
    (ws,) = run_streams(server, 1)
    sentences, cur = [], ""
    for _, m in ws.sent:
        msg = m.get("msg")
        if isinstance(msg, dict):
            cur += msg.get("text", "")
            if msg.get("reset"):
                sentences.append(cur)
                cur = ""
    assert len(sentences) == SENTENCES + 1
    # each sentence restarts the fake decoder, so its text is a prefix of the sequence
    for text in sentences[:SENTENCES]:
        assert text and text == "".join(chr(0x4E00 + i) for i in range(len(text))), text
