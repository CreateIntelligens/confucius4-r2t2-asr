import numpy as np
import soundfile as sf

from audio_io import load_audio


def tone(sr, seconds=0.5):
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_16k_mono_is_passed_through(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, tone(16000), 16000, subtype="FLOAT")
    wav = load_audio(str(path))
    assert wav.dtype == np.float32
    np.testing.assert_allclose(wav, tone(16000), atol=1e-6)


def test_stereo_48k_becomes_16k_mono(tmp_path):
    path = tmp_path / "b.wav"
    mono = tone(48000)
    sf.write(path, np.stack([mono, mono], axis=1), 48000)
    wav = load_audio(str(path))
    assert wav.ndim == 1
    assert abs(len(wav) - 8000) <= 1
    # still the same 440 Hz tone
    spectrum = np.abs(np.fft.rfft(wav))
    assert abs(np.argmax(spectrum) * 16000 / len(wav) - 440) < 5


def test_matches_librosa(tmp_path):
    import librosa

    path = tmp_path / "c.wav"
    sf.write(path, tone(44100), 44100)
    expected, _ = librosa.load(str(path), sr=16000, mono=True)
    np.testing.assert_allclose(load_audio(str(path)), expected, atol=1e-5)
