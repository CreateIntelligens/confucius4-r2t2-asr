import numpy as np

from r2t2 import R2T2ASRModel
from r2t2.r2t2_asr import ASRStreamingState

SR = 16000


def state_with(chunk_samples, texts, pending=0):
    return ASRStreamingState(
        unfixed_chunk_num=0,
        unfixed_token_num=1,
        chunk_size_sec=0.16,
        chunk_size_samples=2560,
        chunk_id=len(texts),
        buffer=np.zeros((0,), dtype=np.float32),
        audio_accum=np.zeros(sum(chunk_samples) + pending, dtype=np.float32),
        prompt_raw="",
        context="",
        force_language="Chinese",
        language="Chinese",
        text="",
        _raw_decoded="",
        chunk_text=list(texts),
        last_fixed_text="".join(texts),
        _first_chunk_discarded=False,
        chunk_samples=list(chunk_samples),
        pending_samples=pending,
        lang_meta="",
    )


def test_window_under_limit_is_untouched():
    st = state_with([2560] * 50, ["字"] * 50)
    R2T2ASRModel._trim_streaming_window(st)
    assert st.audio_accum.shape[0] == 50 * 2560
    assert len(st.chunk_text) == 50


def test_uniform_chunks_drop_exactly_eight_seconds():
    n = 16 * SR // 2560 + 1
    st = state_with([2560] * n, [str(i % 10) for i in range(n)])
    R2T2ASRModel._trim_streaming_window(st)
    assert st.audio_accum.shape[0] == (n - 50) * 2560
    assert len(st.chunk_text) == len(st.chunk_samples) == n - 50


def test_mixed_chunk_sizes_keep_text_aligned_with_audio():
    sizes = [10240, 2560, 20480] * 10
    texts = [f"<{i}>" for i in range(len(sizes))]
    st = state_with(sizes, texts, pending=2560)
    R2T2ASRModel._trim_streaming_window(st)
    # whatever audio is left is exactly the audio of the chunks whose text is left
    assert st.audio_accum.shape[0] == sum(st.chunk_samples) + st.pending_samples
    assert st.chunk_text == texts[len(texts) - len(st.chunk_text):]
    assert sum(sizes) + 2560 - st.audio_accum.shape[0] >= 8 * SR


def test_audio_is_bounded_even_before_any_text_is_decoded():
    st = state_with([], [], pending=17 * SR)
    R2T2ASRModel._trim_streaming_window(st)
    assert st.audio_accum.shape[0] == 9 * SR
    assert st.pending_samples == 9 * SR
