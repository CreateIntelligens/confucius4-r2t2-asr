"""Decode an uploaded audio file to 16 kHz mono float32."""

import numpy as np
import soundfile as sf
import soxr

TARGET_SAMPLE_RATE = 16000


def load_audio(path: str) -> np.ndarray:
    """Read ``path`` as 16 kHz mono float32.

    soundfile 加 soxr 直接解碼與重取樣，結果與 librosa.load 相同（它內部也是 soxr），
    但省掉 librosa 第一次呼叫的初始化：在容器內每次啟動都要重付，慢的機器上曾讓
    服務啟動後的第一個上傳多等 27 秒。soundfile 讀不了的格式（如 m4a）才退回 librosa。
    """
    try:
        wav, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception:
        import librosa

        wav, _ = librosa.load(path, sr=TARGET_SAMPLE_RATE, mono=True)
        return wav.astype(np.float32, copy=False)
    wav = wav.mean(axis=1)
    if sr != TARGET_SAMPLE_RATE:
        wav = soxr.resample(wav, sr, TARGET_SAMPLE_RATE)
    return np.ascontiguousarray(wav, dtype=np.float32)
