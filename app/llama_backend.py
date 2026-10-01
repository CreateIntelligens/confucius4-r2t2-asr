"""llama.cpp decoding behind the same model interface as the other backends.

The audio encoder stays in PyTorch (the weights the model was trained with);
only token generation moves to llama.cpp, which needs neither vLLM nor a
prebuilt wheel for the host's GPU: it is compiled from source for whatever
architecture the image is built on.

The upstream ``r2t2_llama`` hybrid route does the same split but borrows its
encoder class from the vLLM package, so it cannot be imported where vLLM is
not installed. This module uses the Transformers encoder instead.
"""

import glob
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

from qwen_asr.inference.utils import (
    normalize_language_name,
    parse_asr_output,
    validate_language,
)
from r2t2 import R2T2ASRModel

log = logging.getLogger("r2t2-service")

GGUF_REPO_ID = "mradermacher/Confucius4-R2T2-GGUF"
# Q8 解碼器實測與原始權重的辨識結果只差在標點；projector 用 f16。
GGUF_FILES = ("Confucius4-R2T2.Q8_0.gguf", "Confucius4-R2T2.mmproj-f16.gguf")
N_CTX = 8192
N_BATCH = 2048


def import_native():
    """Return the compiled ``Qwen3ASRNative`` class, or None if it is not usable here."""
    try:
        # built into the image for its own architecture and Python
        from qwen3asr_native import Qwen3ASRNative

        return Qwen3ASRNative
    except ImportError:
        pass
    try:
        # prebuilt artifacts shipped in the repository (x86_64, CPython 3.12)
        from r2t2_llama.native.qwen3asr_native import Qwen3ASRNative

        return Qwen3ASRNative
    except ImportError:
        return None


def find_gguf(gguf_dir: str) -> Optional[Tuple[str, str]]:
    """Return (decoder, projector) GGUF paths, or None unless exactly one of each exists."""
    found = sorted(glob.glob(os.path.join(gguf_dir, "*.gguf")))
    projectors = [p for p in found if "mmproj" in os.path.basename(p)]
    decoders = [p for p in found if "mmproj" not in os.path.basename(p)]
    if len(projectors) != 1 or len(decoders) != 1:
        return None
    return decoders[0], projectors[0]


def ensure_gguf(gguf_dir: str, auto_download: bool) -> Optional[Tuple[str, str]]:
    paths = find_gguf(gguf_dir)
    if paths or not auto_download:
        return paths
    from huggingface_hub import hf_hub_download

    log.info("下載 llama.cpp 用的 GGUF 模型 %s 到 %s ...", GGUF_REPO_ID, gguf_dir)
    os.makedirs(gguf_dir, exist_ok=True)
    for name in GGUF_FILES:
        hf_hub_download(GGUF_REPO_ID, name, local_dir=gguf_dir)
    return find_gguf(gguf_dir)


class R2T2LlamaModel(R2T2ASRModel):
    """R2T2 streaming with a PyTorch audio encoder and a llama.cpp decoder."""

    @classmethod
    def load(cls, model_dir: str, decoder_gguf: str, projector_gguf: str, native_cls) -> "R2T2LlamaModel":
        base = R2T2ASRModel.from_pretrained(model_dir, device_map="cuda", dtype=torch.bfloat16)
        thinker = getattr(base.model, "thinker", base.model)
        # 文字解碼交給 llama.cpp，PyTorch 這邊只留聲學編碼器，省下約 3 GB 顯存。
        if hasattr(thinker, "model"):
            del thinker.model
        if hasattr(thinker, "lm_head"):
            del thinker.lm_head
        torch.cuda.empty_cache()

        self = cls(
            backend="llama",
            model=base.model,
            processor=base.processor,
            max_inference_batch_size=1,
            max_new_tokens=512,
        )
        self._thinker = thinker
        threads = max(1, min(8, os.cpu_count() or 1))
        # llama.cpp 的 context 必須固定由同一條執行緒建立與呼叫。服務把每次推論丟給
        # 執行緒池，多路連線時會輪到不同執行緒，直接呼叫會讓行程 segfault。
        self._native_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama")
        self._native = self._native_thread.submit(
            native_cls, decoder_gguf, projector_gguf, N_CTX, N_BATCH, threads, True, -1
        ).result()
        return self

    @torch.no_grad()
    def _encode(self, audio: np.ndarray) -> np.ndarray:
        feats = self.processor.feature_extractor(
            np.asarray(audio, dtype=np.float32).reshape(-1),
            sampling_rate=16000,
            return_attention_mask=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        device, dtype = self.model.device, self.model.dtype
        out = self._thinker.get_audio_features(
            feats["input_features"].to(device, dtype),
            feature_attention_mask=feats["attention_mask"].to(device),
        )
        return np.ascontiguousarray(
            out.float().cpu().numpy().reshape(-1, out.shape[-1]), dtype=np.float32
        )

    def _generate_step(self, prompt: str, audio: np.ndarray, max_new_tokens=None) -> str:
        result = self._native_thread.submit(
            self._native.generate_once_with_embedding,
            self._encode(audio),
            prompt,
            int(max_new_tokens or self.max_new_tokens),
        ).result()
        return result["text"]

    def transcribe(
        self,
        audio: Union[Any, List[Any]],
        context: Union[str, List[str]] = "",
        language: Optional[Union[str, List[Optional[str]]]] = None,
        return_time_stamps: bool = False,
    ):
        """One-pass decode of each ``(waveform, 16000)`` item."""
        if return_time_stamps:
            raise ValueError("timestamps are not available with the llama backend")
        items = audio if isinstance(audio, list) else [audio]
        contexts = context if isinstance(context, list) else [context] * len(items)
        languages = language if isinstance(language, list) else [language] * len(items)

        results = []
        for (wav, _), ctx, lang in zip(items, contexts, languages):
            force_language = None
            if lang is not None and str(lang).strip():
                force_language = normalize_language_name(str(lang))
                validate_language(force_language)
            prompt = self._build_text_prompt(context=ctx or "", force_language=force_language)
            raw = self._generate_step(prompt, wav, self.max_new_tokens)
            detected, text = parse_asr_output(raw, user_language=force_language)
            results.append(SimpleNamespace(language=detected, text=text, time_stamps=None))
        return results
