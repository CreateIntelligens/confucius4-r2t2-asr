"""Simplified -> Traditional (Taiwan) conversion of recognised text."""

from typing import Optional

from opencc import OpenCC

OUTPUT_SCRIPTS = ("traditional", "simplified")
DEFAULT_OUTPUT_SCRIPT = "simplified"

# s2twp：簡體轉台灣繁體，並換成台灣慣用語（软件→軟體、鼠标→滑鼠）。模型只會輸出簡體。
_CONVERTER = OpenCC("s2twp")


def resolve_output_script(value: Optional[str]) -> str:
    """Validate a client's ``output_script``; absent means simplified."""
    script = (value or DEFAULT_OUTPUT_SCRIPT).strip().lower()
    if script not in OUTPUT_SCRIPTS:
        raise ValueError("Unsupported output script")
    return script


def convert_text(text: str, output_script: str) -> str:
    return _CONVERTER.convert(text) if output_script == "traditional" and text else text


class StreamingConverter:
    """Converts append-only deltas of one sentence.

    慣用語要看到整個詞才轉得對，所以每次都轉「這一句累積到目前的全文」，再送出
    新增的部分：软、件 分兩次到也會得到 軟體。若後到的字改變了已經送出的字
    （鼠 之後來了 标，整詞變成 滑鼠），串流只增不改，收不回來，這時只轉新增的字；
    句末的 final_text 會用整句再轉一次，那才是定稿。
    """

    def __init__(self, output_script: str) -> None:
        self._script = output_script
        self._source = ""
        self._sent = ""

    def push(self, delta: str) -> str:
        if self._script != "traditional" or not delta:
            return delta
        self._source += delta
        converted = _CONVERTER.convert(self._source)
        if converted.startswith(self._sent):
            out = converted[len(self._sent):]
        else:
            out = _CONVERTER.convert(delta)
        self._sent += out
        return out

    def reset(self) -> None:
        self._source = ""
        self._sent = ""
