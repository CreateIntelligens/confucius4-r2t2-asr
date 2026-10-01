"""Per-connection streaming VAD built on one shared FireRedVAD model."""

import logging
import os
from typing import Optional

import numpy as np
import torch

log = logging.getLogger("r2t2-service")

VAD_REPO_ID = "FireRedTeam/FireRedVAD"
VAD_SUBDIR = "Stream-VAD"


class StreamVad:
    """Speech start/end detector for one audio stream."""

    def __init__(self, vad) -> None:
        self._vad = vad

    def detect(self, chunk: np.ndarray):
        """Return (speech_started, speech_ended) for a float32 chunk."""
        pcm = (chunk * 32768.0).clip(-32768, 32767).astype(np.int16)
        with torch.inference_mode():
            results = self._vad.detect_chunk(pcm)
        started = any(r.is_speech_start for r in results)
        ended = any(r.is_speech_end for r in results)
        return started, ended


class VadFactory:
    """Loads the VAD weights once and hands out independent detectors.

    FireRedStreamVad keeps its feature, model-cache and state-machine state
    on the instance, so sharing one instance between connections mixes their
    audio and breaks end-of-speech detection. Only the weights are shared.
    """

    def __init__(self, model_dir: str) -> None:
        from fireredvad import FireRedStreamVad, FireRedStreamVadConfig

        # 停頓 60 幀（約 600ms）視為一句結束；閥值 0.4 偏寬鬆，寧可多收不要漏字。
        self._config = FireRedStreamVadConfig(
            use_gpu=False,
            smooth_window_size=5,
            speech_threshold=0.4,
            pad_start_frame=5,
            min_speech_frame=8,
            max_speech_frame=2000,
            min_silence_frame=60,
            chunk_max_frame=30000,
        )
        self._model_dir = model_dir
        self._cls = FireRedStreamVad
        self._shared = FireRedStreamVad.from_pretrained(model_dir, self._config)
        self._shared.vad_model.eval()

    def new(self) -> StreamVad:
        from fireredvad.core.audio_feat import AudioFeat
        from fireredvad.core.stream_vad_postprocessor import StreamVadPostprocessor

        cfg = self._config
        vad = self._cls(
            AudioFeat(os.path.join(self._model_dir, "cmvn.ark")),
            self._shared.vad_model,
            StreamVadPostprocessor(
                cfg.smooth_window_size,
                cfg.speech_threshold,
                cfg.pad_start_frame,
                cfg.min_speech_frame,
                cfg.max_speech_frame,
                cfg.min_silence_frame,
            ),
            cfg,
        )
        return StreamVad(vad)


def load_vad_factory(vad_dir: str, auto_download: bool) -> Optional[VadFactory]:
    """Return a factory, or None when VAD cannot be set up (streaming still works)."""
    model_dir = os.path.join(vad_dir, VAD_SUBDIR)
    try:
        if not os.path.exists(os.path.join(model_dir, "model.pth.tar")):
            if not auto_download:
                log.warning("找不到 VAD 模型 (%s) 且 AUTO_DOWNLOAD=0，串流將不自動斷句", model_dir)
                return None
            from huggingface_hub import snapshot_download

            log.info("下載串流 VAD 模型 %s 到 %s ...", VAD_REPO_ID, vad_dir)
            snapshot_download(
                repo_id=VAD_REPO_ID,
                allow_patterns=[f"{VAD_SUBDIR}/*"],
                local_dir=vad_dir,
            )
        return VadFactory(model_dir)
    except Exception as exc:
        log.warning("串流 VAD 初始化失敗，串流將不自動斷句: %s", exc)
        return None
