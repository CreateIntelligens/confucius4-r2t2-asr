import pytest

from textproc import (
    clean_transcript,
    detect_hallucination,
    keep_streamed_ending,
    normalize_language,
)


@pytest.mark.parametrize(
    "code, expected",
    [
        ("zhen", "Chinese"),
        ("ZHEN", "Chinese"),
        ("zh", "Chinese"),
        ("zh-TW", "Chinese"),
        ("Chinese", "Chinese"),
        ("en", "English"),
        ("english", "English"),
        ("yue", "Cantonese"),
        ("japanese", "Japanese"),
        ("auto", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_language(code, expected):
    assert normalize_language(code) == expected


def test_stream_default_applies_only_when_language_missing():
    assert normalize_language(None, "zhen") == "Chinese"
    assert normalize_language("", "zhen") == "Chinese"
    assert normalize_language("en", "zhen") == "English"
    # asking for detection explicitly must win over the default
    assert normalize_language("auto", "zhen") is None


def test_clean_transcript_drops_segment_marker_and_repeats():
    assert clean_transcript("我想了解 hippo 的|") == "我想了解 hippo 的"
    assert clean_transcript("謝謝大家" * 6) == "謝謝大家"


def test_detect_hallucination():
    assert detect_hallucination("今天天氣不錯" + "好的" * 6)[0]
    assert not detect_hallucination("今天天氣不錯，希望你有美好的一天")[0]
    assert not detect_hallucination("")[0]


def test_keep_streamed_ending_restores_a_dropped_last_word():
    streamed = "Could you tell me the maximum flow rate of this pump?"
    assert keep_streamed_ending("Could you tell me the maximum flow rate of this", streamed) == streamed
    # leading junk the stream committed early must not come back
    assert keep_streamed_ending("請問揚程是", "好請問揚程是多少") == "請問揚程是多少"


def test_keep_streamed_ending_leaves_a_complete_final_alone():
    assert keep_streamed_ending("今天下午開會。", "好今天下午開會") == "今天下午開會。"
    assert keep_streamed_ending("今天下午開會。", "今天下午開會。") == "今天下午開會。"
    # texts that do not line up: trust the one-pass result
    assert keep_streamed_ending("我想了解 DIVA PRO", "我想了解diypro的差別") == "我想了解 DIVA PRO"
    assert keep_streamed_ending("", "好") == ""
