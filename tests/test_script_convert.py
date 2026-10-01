import pytest

from script_convert import StreamingConverter, convert_text, resolve_output_script


def test_convert_text():
    assert convert_text("请问软件在哪里", "traditional") == "請問軟體在哪裡"
    assert convert_text("请问软件在哪里", "simplified") == "请问软件在哪里"
    assert convert_text("", "traditional") == ""


def test_resolve_output_script():
    assert resolve_output_script(None) == "simplified"
    assert resolve_output_script("Traditional") == "traditional"
    with pytest.raises(ValueError):
        resolve_output_script("pinyin")


def stream(deltas, script="traditional"):
    conv = StreamingConverter(script)
    return [conv.push(d) for d in deltas]


def test_phrase_split_across_deltas_is_still_localised():
    # converting each delta on its own would give 軟 + 件
    assert "".join(stream(["请问", "软", "件", "在哪里"])) == "請問軟體在哪裡"


def test_output_is_append_only_when_a_later_char_rewrites_an_earlier_one():
    # 鼠标 as a whole is 滑鼠, but 鼠 has already been sent and cannot be taken back
    out = stream(["鼠", "标"])
    assert out[0] == "鼠"
    assert out[1] == "標"


def test_simplified_passes_through_untouched():
    assert stream(["软", "件"], "simplified") == ["软", "件"]


def test_reset_starts_a_new_sentence():
    conv = StreamingConverter("traditional")
    conv.push("软")
    conv.reset()
    assert conv.push("件") == "件"
