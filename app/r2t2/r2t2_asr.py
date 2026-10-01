# coding=utf-8
# Copyright 2026 The NetEase Youdao team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any, Tuple, List, Optional, Union

import numpy as np
import torch,string,math,re

from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
from qwen_asr.inference.utils import (
    SAMPLE_RATE,
    _ASR_TEXT_TAG,
    _LANG_PREFIX,
    normalize_language_name,
    parse_asr_output,
    validate_language,
)


_EN2ZH_PUNCT = {',': '，', '.': '。', '!': '！', '?': '？', ';': '；', ':': '：', '(': '（', ')': '）'}
_ZH2EN_PUNCT = {v: k for k, v in _EN2ZH_PUNCT.items()}
# Tokens the model needs for its own "language X<asr_text>" header.
_LANG_TAG_TOKENS = 4
_ALL_PUNCT_PAT = re.compile(r'[,\.!?;:()\uff0c\u3002\uff01\uff1f\uff1b\uff1a\uff08\uff09]')


def _normalize_punct_by_context(text: str) -> str:
    """Normalize punctuation based on the character type before each mark:
    - If the preceding character is Chinese, replace it with Chinese punctuation.
    - If the preceding character is an English character or digit, replace it with English punctuation.
    - Preserve the original mark in all other cases.
    """
    def _replace(m):
        punct = m.group()
        pos = m.start()
        prev_char = ""
        for i in range(pos - 1, -1, -1):
            if not text[i].isspace():
                prev_char = text[i]
                break
        if not prev_char:
            return punct
        if '\u4e00' <= prev_char <= '\u9fff':
            return _EN2ZH_PUNCT.get(punct, punct)
        elif prev_char.isascii() and (prev_char.isalnum() or prev_char in '"\''):
            return _ZH2EN_PUNCT.get(punct, punct)
        return punct
    return _ALL_PUNCT_PAT.sub(_replace, text)

def parse_language_output(
    raw: str,
    user_language: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Parse Qwen3-ASR raw output into (language, text).

    Cases:
      - With tag: "language Chinese<asr_text>...."
      - With newlines: "language Chinese\\n...\\n<asr_text>...."
      - No tag: treat whole string as text.
      - "language None<asr_text>": treat as empty audio -> ("", "")

    If user_language is provided, language is forced to user_language and raw is treated as text-only
    (the model is expected to output plain transcription without metadata).

    Args:
        raw: Raw decoded string.
        user_language: Canonical language name if user forced language.

    Returns:
        Tuple[str, str]: (language, text)
    """
    if raw is None:
        return "", ""
    if user_language == "English":
        s = str(raw).rstrip()
    else:
        s = str(raw).strip()
    if not s:
        return "", ""

    if user_language:
        # user explicitly forced language => model output is treated as pure text
        return user_language, s

    meta_part = s
    text_part = ""
    has_tag = _ASR_TEXT_TAG in s
    if has_tag:
        meta_part, text_part = s.split(_ASR_TEXT_TAG, 1)
    else:
        # no tag => pure text
        return "", s.strip()

    meta_lower = meta_part.lower()

    # empty audio heuristic
    if "language none" in meta_lower:
        t = text_part.strip()
        if not t:
            return "", ""
        # if model still returned something, keep it but language unknown
        return "", t

    # extract "language xxx" from meta
    lang = ""
    for line in meta_part.splitlines():
        line = line.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(_LANG_PREFIX):
            val = line[len(_LANG_PREFIX):].strip()
            if val:
                lang = normalize_language_name(val)
            break

    return lang, text_part.strip()

@dataclass
class ASRStreamingState:
    """
    Streaming ASR state for one audio stream (single utterance).

    Attributes:
        unfixed_chunk_num (int):
            For the first N chunks, do not use previous ASR result as prefix prompt (reset prefix to "").
        unfixed_token_num (int):
            When chunk_id >= unfixed_chunk_num, rollback the last K tokens from the accumulated text
            before using it as prefix prompt, to reduce boundary jitter.
        chunk_size_sec (float):
            Chunk size in seconds. Audio will be fed to the model in increments of this length.
        chunk_size_samples (int):
            Chunk size in samples at 16kHz (derived from chunk_size_sec).
        chunk_id (int):
            Current chunk index (0-based).
        buffer (np.ndarray):
            Buffered PCM samples that are not yet consumed into a full chunk.
        audio_accum (np.ndarray):
            Accumulated audio from the beginning of the stream up to current time (no padding).
        prompt_raw (str):
            Base prompt generated by chat template (with generation prompt), without appended prefix text.
        context (str):
            Context string.
        force_language (Optional[str]):
            If provided, force output to be text-only by appending "language X<asr_text>" in prompt_raw,
            consistent with non-streaming transcribe().
        language (str):
            Latest parsed language (updated after each chunk decode). Empty if unknown/silent.
        text (str):
            Latest parsed transcription text (updated after each chunk decode).
        _raw_decoded (str):
            Internal accumulated decoded raw text (before parse_asr_output normalization).
            Used for rollback/token trimming and as prefix for prompting.
    """
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_size_sec: float
    chunk_size_samples: int

    chunk_id: int
    buffer: np.ndarray
    audio_accum: np.ndarray

    prompt_raw: str
    context: str
    force_language: Optional[str]

    language: str
    text: str
    _raw_decoded: str

    chunk_text: List
    last_fixed_text: str
    _first_chunk_discarded: bool
    chunk_samples: List
    pending_samples: int
    lang_meta: str

class R2T2ASRModel(Qwen3ASRModel):
    def __init__(
        self,
        backend: str,
        model: Any,
        processor: Any,
        sampling_params: Optional[Any] = None,
        forced_aligner: Optional[Any] = None,
        max_inference_batch_size: int = -1,
        max_new_tokens: int = 512,
    ):

        super().__init__(
            backend=backend,
            model=model,
            processor=processor,
            sampling_params=sampling_params,
            forced_aligner=forced_aligner,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

    @torch.no_grad()
    def transcribe(
        self,
        audio: Union[Any, List[Any]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ):
        return super().transcribe(
            audio=audio,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )

    @torch.no_grad()
    def _generate_step(
        self,
        prompt: str,
        audio: np.ndarray,
        max_new_tokens=None,
    ) -> str:
        """Run one decode over ``audio`` continuing from ``prompt``.

        Streaming re-feeds the whole audio window with the already fixed text
        appended to the prompt, so the only backend-specific part is this
        single generate call.
        """
        if self.backend == "vllm":
            sampling_params = self.sampling_params
            if max_new_tokens is not None:
                from vllm import SamplingParams

                sampling_params = SamplingParams(
                    temperature=0.0,
                    max_tokens=int(max_new_tokens),
                    skip_special_tokens=True,
                )
            inp = {"prompt": prompt, "multi_modal_data": {"audio": [audio]}}
            outputs = self.model.generate([inp], sampling_params=sampling_params, use_tqdm=False)
            return outputs[0].outputs[0].text

        inputs = self.processor(text=[prompt], audio=[audio], return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device).to(self.model.dtype)
        text_ids = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens or self.max_new_tokens),
        )
        decoded = self.processor.batch_decode(
            text_ids.sequences[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return decoded[0]

    @staticmethod
    def _trim_streaming_window(state: "ASRStreamingState") -> None:
        """Keep the rolling window under 16 s by dropping the oldest ~8 s.

        The text of the dropped chunks leaves the prefix together with their
        audio. Trimming by the samples each chunk actually covered (instead
        of a fixed chunk count) keeps audio and prefix aligned when the
        caller feeds chunks of varying size.
        """
        max_samples = 16 * SAMPLE_RATE
        discard_samples = 8 * SAMPLE_RATE
        if state.audio_accum.shape[0] <= max_samples:
            return
        dropped = 0
        n = 0
        while n < len(state.chunk_samples) and dropped < discard_samples:
            dropped += state.chunk_samples[n]
            n += 1
        if dropped == 0:
            # Nothing decoded yet (e.g. language still undetected on silence):
            # there is no prefix to keep aligned, only audio to bound.
            dropped = discard_samples
            state.pending_samples = max(0, state.pending_samples - dropped)
        state.audio_accum = state.audio_accum[dropped:]
        state.chunk_text = state.chunk_text[n:]
        state.chunk_samples = state.chunk_samples[n:]
        state._first_chunk_discarded = True

    def init_streaming_state(
        self,
        context: str = "",
        language: Optional[str] = None,
        unfixed_chunk_num: int = 2,
        unfixed_token_num: int = 5,
        chunk_size_sec: float = 2.0,
    ) -> ASRStreamingState:
        """
        Initialize streaming ASR state for a single stream.

        Notes:
            - Works on both the vLLM and the Transformers backend.
            - Streaming ASR does NOT support timestamps (forced aligner is not used).
            - Batch inference is NOT supported.

        Args:
            context:
                Context string.
            language:
                Optional forced language. If provided, it must be in supported languages.
                Same behavior as transcribe(): forces text-only output via prompt suffix.
            unfixed_chunk_num:
                For the first N chunks, do not use previous output as prefix prompt (reset prefix to "").
            unfixed_token_num:
                Roll back the last K tokens from accumulated output when using it as prefix prompt
                after unfixed_chunk_num.
            chunk_size_sec:
                Chunk size in seconds (audio is 16k PCM). The function will internally convert it
                to sample count at 16kHz.

        Returns:
            ASRStreamingState: Mutable state object to be passed to streaming_transcribe() and
            finish_streaming_transcribe().

        Raises:
            ValueError:
                - If language is not supported.
                - If chunk_size_sec <= 0.
                - If forced language is invalid (same validation rules as transcribe()).
        """
        if chunk_size_sec is None or float(chunk_size_sec) <= 0:
            raise ValueError(f"chunk_size_sec must be > 0, got: {chunk_size_sec}")

        force_language = None
        if language is not None and str(language).strip() != "":
            ln = normalize_language_name(str(language))
            validate_language(ln)
            force_language = ln

        chunk_size_samples = int(round(float(chunk_size_sec) * SAMPLE_RATE))
        chunk_size_samples = max(1, chunk_size_samples)

        prompt_raw = self._build_text_prompt(context=context, force_language=force_language)

        return ASRStreamingState(
            unfixed_chunk_num=int(unfixed_chunk_num),
            unfixed_token_num=int(unfixed_token_num),
            chunk_size_sec=float(chunk_size_sec),
            chunk_size_samples=int(chunk_size_samples),
            chunk_id=0,
            buffer=np.zeros((0,), dtype=np.float32),
            audio_accum=np.zeros((0,), dtype=np.float32),
            prompt_raw=prompt_raw,
            context=context or "",
            force_language=force_language,
            language="",
            text="",
            _raw_decoded="",
            chunk_text=[],
            last_fixed_text="",
            _first_chunk_discarded=False,
            chunk_samples=[],
            pending_samples=0,
            lang_meta="",
        )

    def streaming_transcribe(
        self,
        pcm16k: np.ndarray,
        state: ASRStreamingState,
        max_new_tokens=None,
        rollback_punctuation: bool = False,
    ) -> Tuple[str, str]:
        """
        Streaming ASR decode step.

        This function accepts an arbitrary-length 16k PCM float numpy array (mono).
        It buffers incoming samples, and whenever enough samples are accumulated to form one
        full chunk (chunk_size_sec), it runs one incremental decode step and updates:

          - state.language
          - state.text

        The caller only needs to keep passing audio to this function and read state.language/state.text.

        Implementation details:
            - Each time a new chunk is ready, we append it to audio_accum and re-feed *all* audio seen
              so far to the model (no padding).
            - We update the prompt as: state.prompt_raw + prefix_text
            - Prefix rollback strategy:
                * If chunk_id < unfixed_chunk_num: prefix_text = ""
                * Else: rollback last unfixed_token_num tokens from previously accumulated decoded text.

        Notes:
            - Works on both the vLLM and the Transformers backend.
            - No timestamps.
            - Single stream only (no batching).

        Args:
            pcm16k:
                16kHz mono PCM waveform (np.ndarray). Length can be any non-negative integer.
                dtype can be float32/float64/int16; it will be converted to float32.
            state:
                Streaming state returned by init_streaming_state().

        Returns:
            Tuple[str, str]: A tuple containing the latest transcription text
            (``state.text``) and the text finalized for the current update
            (``fixed_text``). The state object is mutated in place.

        Raises:
            ValueError:
                If state is invalid.
        """
        if state is None:
            raise ValueError("state must not be None. Call init_streaming_state() first.")
        if pcm16k is None:
            raise ValueError("pcm16k must not be None.")

        # Ensure 1D mono
        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)

        # Convert to float32 PCM in [-1, 1] if int16 provided
        if x.dtype == np.int16:
            x = (x.astype(np.float32) / 32768.0)
        else:
            x = x.astype(np.float32, copy=False)

        # Append to buffer
        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        # Consume full chunks
        fixed_text = ""  # Initialize to avoid UnboundLocalError if the loop does not run
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]

            # Accumulate audio (re-feed from start, no padding)
            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)

            # Build prefix with rollback strategy
            prefix = ""
            if state.chunk_id < state.unfixed_chunk_num:
                prefix = ""
            else:
                state._raw_decoded = state._raw_decoded.split("|")[0]
                cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
                if rollback_punctuation == True:
                    _punct = "，。！？、；：.!?;:"
                    raw_stripped = state._raw_decoded.strip()
                    if raw_stripped and raw_stripped[-1] in _punct:
                        k = 0
                    else:
                        k = int(state.unfixed_token_num)
                else:
                    k = int(state.unfixed_token_num)
                while True:
                    end_idx = max(0, len(cur_ids) - k)
                    prefix = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                    if '\ufffd' not in prefix:
                        break
                    else:
                        if end_idx == 0:
                            prefix = ""
                            break
                        k += 1

            prefix = prefix.split("|")[0]
            
            prompt = state.prompt_raw + prefix

            gen_text = self._generate_step(prompt, state.audio_accum, max_new_tokens)
            gen_text = _normalize_punct_by_context(gen_text).replace('\ufffd', '')
            #print(f"prefix={prefix}")
            state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
            #print(f"state._raw_decoded={state._raw_decoded}")

            lang = None
            if state.force_language == None:
                lang,_ = parse_language_output(state._raw_decoded, user_language=state.force_language)

            if state.force_language == "Chinese" or lang == "Chinese":
                # Remove extra spaces between adjacent Chinese characters; preserve English spaces in mixed-language text
                state._raw_decoded = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', state._raw_decoded)  
            lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)

            has_tag = "<asr_text>" in state._raw_decoded
            if has_tag:
                state._raw_decoded = state._raw_decoded.split("<asr_text>")[0] + "<asr_text>" + txt
            else:
                state._raw_decoded = txt
            
            state._raw_decoded = state._raw_decoded.split("|")[0]
            cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
            if rollback_punctuation == True:
                _punct = "，。！？、；：,.!?;:"
                raw_stripped = state._raw_decoded.strip()
                if raw_stripped and raw_stripped[-1] in _punct:
                    k = 0
                else:
                    k = int(state.unfixed_token_num)
            else:
                k = int(state.unfixed_token_num)

            has_tag = "<asr_text>" in state._raw_decoded
            if has_tag and state._raw_decoded.split('<asr_text>')[1] == "":
                k = 0
            while True:
                end_idx = max(0, len(cur_ids) - k)
                fixed_text = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                if '\ufffd' not in fixed_text:
                    break
                else:
                    if end_idx == 0:
                        fixed_text = ""
                        break
                    k += 1
            has_tag = "<asr_text>" in fixed_text
            if has_tag:
                meta_part, fixed_text  = fixed_text.split("<asr_text>", 1)
            fixed_text = fixed_text.split("|")[0]
            
            has_tag = "<asr_text>" in state._raw_decoded
            if not has_tag and state.force_language == None:
                state.text = ""
                fixed_text = ""
                continue
  
            state.language = lang
            state.text = txt.split("|")[0]

            state.chunk_id += 1

        return state.text,fixed_text

    def finish_streaming_transcribe(
        self,
        state: ASRStreamingState,
        max_new_tokens=None,
    ) -> str:
        """
        Finish streaming ASR.

        This function flushes the remaining buffered audio in state.buffer (tail audio).
        It sends the remaining samples to the model even if shorter than chunk_size_sec,
        without padding. Then it updates state.language/state.text one last time.

        Notes:
            - Works on both the vLLM and the Transformers backend.
            - No timestamps.
            - Single stream only.

        Args:
            state:
                Streaming state returned by init_streaming_state().
            max_new_tokens:
                Optional maximum number of tokens to generate for the final
                decode.

        Returns:
            str: The latest transcription text (``state.text``). The state
            object is mutated in place and can be reused for subsequent
            streaming calls.

        Raises:
            ValueError:
                If state is invalid.
        """
        if state is None:
            raise ValueError("state must not be None.")

        # If no remaining buffer, still return state as-is.
        if state.buffer is None or state.buffer.shape[0] == 0:
            return state.text

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        # Append tail to accumulated audio
        if state.audio_accum.shape[0] == 0:
            state.audio_accum = tail
        else:
            state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)

        # Prefix rollback strategy (same as per-chunk)
        prefix = ""
        if state.chunk_id < state.unfixed_chunk_num:
            prefix = ""
        else:
            cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
            end_idx = max(1, len(cur_ids) - int(state.unfixed_token_num))
            prefix = self.processor.tokenizer.decode(cur_ids[:end_idx])

        prefix = prefix.split("|")[0]
        prompt = state.prompt_raw + prefix
        gen_text = self._generate_step(prompt, state.audio_accum, max_new_tokens)
        gen_text = _normalize_punct_by_context(gen_text).replace('\ufffd', '')

        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        state._raw_decoded = state._raw_decoded.split("|")[0]
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)

        state.language = lang
        state.text = txt.split("|")[0]
        state.chunk_id += 1
        print(f"finish state.text={state.text}")
        return state.text

    def streaming_transcribe_no_reset(
        self,
        pcm16k: np.ndarray,
        state: ASRStreamingState,
        max_new_tokens=None,
        rollback_punctuation: bool = False,
    ) -> Tuple[str, str]:
        """Streaming ASR decode step without resetting accumulated context.

        This variant keeps a rolling audio window and the finalized text
        accumulated in ``state.last_fixed_text``. It buffers incoming samples,
        decodes each complete chunk, and updates the streaming state in place.

        Notes:
            - Works on both the vLLM and the Transformers backend.
            - No timestamps.
            - Single stream only (no batching).

        Args:
            pcm16k:
                16kHz mono PCM waveform (np.ndarray). Length can be any
                non-negative integer. It is converted to float32 internally.
            state:
                Streaming state returned by init_streaming_state().
            max_new_tokens:
                Optional maximum number of tokens to generate for each chunk.
            rollback_punctuation:
                If True, do not roll back tokens when the current output ends
                with punctuation.

        Returns:
            Tuple[str, str]: A tuple containing the latest transcription text
            (``state.text``) and the accumulated finalized transcription text
            (``state.last_fixed_text``). The state object is mutated in place.

        Raises:
            ValueError:
                If state is invalid or pcm16k is None.
        """
        if state is None:
            raise ValueError("state must not be None. Call init_streaming_state() first.")
        if pcm16k is None:
            raise ValueError("pcm16k must not be None.")

        # Ensure 1D mono
        x = np.asarray(pcm16k)
        if x.ndim != 1:
            x = x.reshape(-1)

        # Convert to float32 PCM in [-1, 1] if int16 provided
        if x.dtype == np.int16:
            x = (x.astype(np.float32) / 32768.0)
        else:
            x = x.astype(np.float32, copy=False)

        # Append to buffer
        if x.shape[0] > 0:
            state.buffer = np.concatenate([state.buffer, x], axis=0)

        # Consume full chunks
        fixed_text = ""  # Initialize to avoid UnboundLocalError if the loop does not run
        omit_new_asr_text = ""
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]

            # Accumulate audio (re-feed from start, no padding)
            if state.audio_accum.shape[0] == 0:
                state.audio_accum = chunk
            else:
                state.audio_accum = np.concatenate([state.audio_accum, chunk], axis=0)
            state.pending_samples += int(chunk.shape[0])
            self._trim_streaming_window(state)

            prefix_text = "".join(state.chunk_text)
            prefix = state.lang_meta + prefix_text
            step_tokens = max_new_tokens
            if state.force_language == None and state.lang_meta == "" and max_new_tokens is not None:
                # Until the tag is pinned the model spends its first tokens
                # re-emitting "language X<asr_text>" on every step.
                step_tokens = max_new_tokens + _LANG_TAG_TOKENS
            prefix = prefix.split("|")[0] 

            prompt = state.prompt_raw + prefix

            gen_text = self._generate_step(prompt, state.audio_accum, step_tokens)
            # gen_text = gen_text.replace("#","|")
            gen_text = _normalize_punct_by_context(gen_text).replace('\ufffd', '')

            state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text

            lang = None
            if state.force_language == None:
                lang,_ = parse_language_output(state._raw_decoded, user_language=state.force_language)

            if state.force_language == "Chinese" or lang == "Chinese":
                # Remove extra spaces between adjacent Chinese characters; preserve English spaces in mixed-language text
                state._raw_decoded = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', state._raw_decoded)            
            lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
            has_tag = "<asr_text>" in state._raw_decoded
            if has_tag:
                state._raw_decoded = state._raw_decoded.split("<asr_text>")[0] + "<asr_text>" + txt
            else:
                state._raw_decoded = txt

            state._raw_decoded = state._raw_decoded.split("|")[0]
            cur_ids = self.processor.tokenizer.encode(state._raw_decoded)
            if rollback_punctuation == True:
                _punct = "，。！？、；：,.!?;:"
                raw_stripped = state._raw_decoded.strip()
                if raw_stripped and raw_stripped[-1] in _punct:
                    k = 0
                else:
                    k = int(state.unfixed_token_num)
            else:
                k = int(state.unfixed_token_num)

            has_tag = "<asr_text>" in state._raw_decoded
            if has_tag and state._raw_decoded.split('<asr_text>')[1] == "":
                k = 0
            while True:
                end_idx = max(0, len(cur_ids) - k)
                fixed_text = self.processor.tokenizer.decode(cur_ids[:end_idx]) if end_idx > 0 else ""
                if '\ufffd' not in fixed_text:
                    break
                else:
                    if end_idx == 0:
                        fixed_text = ""
                        break
                    k += 1
            has_tag = "<asr_text>" in fixed_text
            if has_tag:
                meta_part, fixed_text  = fixed_text.split("<asr_text>", 1)
            
            has_tag = "<asr_text>" in state._raw_decoded
            if not has_tag and state.force_language == None:
                state.text = ""
                fixed_text = ""
                continue
  
            state.language = lang
            state.text = txt.split("|")[0]

            state.chunk_id += 1
            prefix_stripped = prefix_text.strip()
            fixed_text_stripped = fixed_text.strip()
            if fixed_text_stripped.startswith(prefix_stripped):
                new_asr_text = fixed_text_stripped[len(prefix_stripped):]
            else:
                new_asr_text = ""
            new_asr_text = new_asr_text.split("|")[0]
            state.chunk_text.append(new_asr_text)
            if state.force_language == None and state.lang_meta == "" and new_asr_text:
                # Pin the tag exactly as the model wrote it (it may well be
                # "language None") once text starts being fixed. Rebuilding it
                # from the parsed language dropped the "None" case, leaving the
                # prefix without a tag and the transcript stuck on its first token.
                state.lang_meta = state._raw_decoded.split("<asr_text>")[0] + "<asr_text>"
            state.chunk_samples.append(state.pending_samples)
            state.pending_samples = 0
            state.last_fixed_text = state.last_fixed_text + new_asr_text


        return state.text, state.last_fixed_text

    def finish_streaming_transcribe_no_reset(
        self,
        state: ASRStreamingState,
        max_new_tokens=None,
    ) -> str:
        """
        Finish streaming ASR without resetting accumulated context.

        This function flushes the remaining buffered audio in state.buffer
        (tail audio). It sends the remaining samples to the model even if they
        are shorter than chunk_size_sec, without padding, and appends any new
        finalized text to ``state.last_fixed_text``.

        Notes:
            - Works on both the vLLM and the Transformers backend.
            - No timestamps.
            - Single stream only.

        Args:
            state:
                Streaming state returned by init_streaming_state().
            max_new_tokens:
                Optional maximum number of tokens to generate for the final
                decode.

        Returns:
            str: The accumulated finalized transcription text
            (``state.last_fixed_text``). The state object is mutated in place.
            If no tail audio remains, the current finalized text is returned
            without issuing another model request.

        Raises:
            ValueError:
                If state is invalid.
        """
        if state is None:
            raise ValueError("state must not be None.")

        # If no remaining buffer, still return state as-is.
        if state.buffer is None or state.buffer.shape[0] == 0:
            # print(f"22text={state.text}")
            return state.last_fixed_text

        tail = state.buffer
        state.buffer = np.zeros((0,), dtype=np.float32)

        # Append tail to accumulated audio
        if state.audio_accum.shape[0] == 0:
            state.audio_accum = tail
        else:
            state.audio_accum = np.concatenate([state.audio_accum, tail], axis=0)
        state.pending_samples += int(tail.shape[0])

        self._trim_streaming_window(state)

        prefix_text = "".join(state.chunk_text)
        prefix = state.lang_meta + prefix_text
        if state.force_language == None and state.lang_meta == "" and max_new_tokens is not None:
            max_new_tokens = max_new_tokens + _LANG_TAG_TOKENS

        prefix = prefix.split("|")[0]    
        prompt = state.prompt_raw + prefix

        gen_text = self._generate_step(prompt, state.audio_accum, max_new_tokens)
   
        gen_text = _normalize_punct_by_context(gen_text).replace('\ufffd', '')
        print(f"finish gen_text={gen_text}")

        state._raw_decoded = (prefix + gen_text) if prefix is not None else gen_text
        state._raw_decoded = state._raw_decoded.split("|")[0]
        lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
        txt = txt.split("|")[0] 
        state.language = lang
        state.text = txt
        state.chunk_id += 1
        prefix_stripped = prefix_text.strip()
        fixed_text_stripped = txt.strip()
        new_asr_text = fixed_text_stripped[len(prefix_stripped):]
        state.chunk_text.append(new_asr_text)
        state.chunk_samples.append(state.pending_samples)
        state.pending_samples = 0
        state.last_fixed_text = state.last_fixed_text + new_asr_text
        return state.last_fixed_text
    


__all__ = ["R2T2ASRModel"]
