"""Text and language helpers shared by the batch and streaming endpoints."""

import re
import string
from typing import List, Optional, Tuple

_PUNCT_CHARS = "，。！？、；：,.!?;:~…·\"'()（）《》—-"
_PUNCT_RE = re.compile(r"[%s\s]+" % re.escape(_PUNCT_CHARS))
_CN_PUNCT = (
    "？！＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～、。〃〄々〆〇〈〉《》「」『』【】"
    "〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏"
)
_STRIP_PUNCT = str.maketrans("", "", string.punctuation + _CN_PUNCT)
_TOKEN_RE = re.compile(r"[a-zA-Z]+|[^a-zA-Z]")

# 「zhen」是有道協議的中英混講代號，模型本身沒有這個語言。
# 混講交給自動判斷會漂到別的語言（帶口音的華語曾被轉成葡萄牙文），
# 而指定 Chinese 時模型仍會照實輸出夾雜的英文，所以對應成 Chinese。
_LANGUAGE_ALIASES = {
    "zhen": "Chinese",
    "chinese": "Chinese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "mandarin": "Chinese",
    "english": "English",
    "en": "English",
    "cantonese": "Cantonese",
    "yue": "Cantonese",
    "japanese": "Japanese",
    "ja": "Japanese",
    "korean": "Korean",
    "ko": "Korean",
    "french": "French",
    "fr": "French",
    "german": "German",
    "de": "German",
    "spanish": "Spanish",
    "es": "Spanish",
}
# 明確要求自動判斷的寫法
AUTO_LANGUAGES = ("auto", "none")


def normalize_language(
    language: Optional[str],
    default: Optional[str] = None,
) -> Optional[str]:
    """Map a client language code to the model's language name.

    Returns None for automatic detection. ``default`` is what an absent or
    empty value means; pass ``"auto"`` explicitly to ask for detection.
    """
    raw = str(language).strip() if language is not None else ""
    if not raw:
        raw = default or ""
    if not raw or raw.lower() in AUTO_LANGUAGES:
        return None
    low = raw.lower()
    if low in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[low]
    return raw[:1].upper() + raw[1:].lower()


_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]")
_KANA_RE = re.compile(r"[\u3040-\u30ff]")
_HAN_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def language_from_text(detected: str, text: str) -> str:
    """Correct the model's language tag using the script it actually wrote.

    The model often tags Mandarin (with or without English mixed in) as
    "language English" while writing the Chinese correctly, so the text is
    the more reliable signal whenever it is not plain Latin script.
    """
    if _HANGUL_RE.search(text):
        return "Korean"
    if _KANA_RE.search(text):
        return "Japanese"
    if _HAN_RE.search(text):
        return detected if detected == "Cantonese" else "Chinese"
    return detected


def detect_and_fix_repetitions(text: str, threshold: int = 4) -> str:
    """
    清洗大模型自回歸解碼時因靜音、雜音引發的跳針與重複幻覺。
    包含單字跳針 (如 '的的的的的的' -> '的') 與詞組跳針 (如 '謝謝大家謝謝大家' -> '謝謝大家')。
    """
    if not text:
        return text

    def fix_char_repeats(s: str, thresh: int) -> str:
        res = []
        i = 0
        n = len(s)
        while i < n:
            count = 1
            while i + count < n and s[i + count] == s[i]:
                count += 1
            if count > thresh:
                res.append(s[i])
                i += count
            else:
                res.append(s[i : i + count])
                i += count
        return "".join(res)

    def fix_pattern_repeats(s: str, thresh: int, max_len: int = 25) -> str:
        n = len(s)
        min_repeat_chars = thresh * 2
        if n < min_repeat_chars:
            return s
        i = 0
        result = []
        while i <= n - min_repeat_chars:
            found = False
            for k in range(1, max_len + 1):
                if i + k * thresh > n:
                    break
                pattern = s[i : i + k]
                if pattern.strip(_PUNCT_CHARS + " \t") == "":
                    continue
                valid = True
                for rep in range(1, thresh):
                    start_idx = i + rep * k
                    if s[start_idx : start_idx + k] != pattern:
                        valid = False
                        break
                if valid:
                    end_index = i + thresh * k
                    while end_index + k <= n and s[end_index : end_index + k] == pattern:
                        end_index += k
                    result.append(pattern)
                    result.append(fix_pattern_repeats(s[end_index:], thresh, max_len))
                    i = n
                    found = True
                    break
            if found:
                break
            else:
                result.append(s[i])
                i += 1
        if not found:
            result.append(s[i:])
        return "".join(result)

    text_cleaned = fix_char_repeats(text, threshold)
    text_cleaned = fix_pattern_repeats(text_cleaned, threshold)
    return text_cleaned


def clean_transcript(text: str) -> str:
    """Final clean-up of a one-pass transcript.

    「|」是 R2T2 模型自己輸出的串流邊界標記，不是內容（官方文件沒有說明；
    實測出現在模型認為最後一個詞還沒聽完的時候）。串流路徑一向丟掉它，
    一次辨識的結果也要比照處理。
    """
    return detect_and_fix_repetitions(text.replace("|", "").strip())


def keep_streamed_ending(final: str, streamed: str) -> str:
    """Put back a sentence ending the one-pass decode left out.

    一次辨識偶爾會在最後一個詞之前就停下來（模型判斷那個詞還沒聽完），
    而串流其實已經聽到並送出了。兩邊文字對得上時，把串流多出來的句尾接回去。
    """
    core = final.rstrip(_PUNCT_CHARS + " \t")
    if not core:
        return final
    idx = streamed.find(core)
    if idx < 0:
        return final
    rest = streamed[idx + len(core):]
    if not rest.strip(_PUNCT_CHARS + " \t"):
        return final
    return core + rest


def detect_hallucination(
    text: str,
    pattern_repeat_thresh: int = 5,
    max_pattern_len: int = 50,
    tail_check_len: int = 256,
) -> Tuple[bool, str]:
    """Tell whether the tail of ``text`` is one pattern repeated over and over."""
    if not text:
        return False, ""
    text = text[-tail_check_len:]
    n = len(text)
    for k in range(1, max_pattern_len + 1):
        if n < k * pattern_repeat_thresh:
            continue
        pattern = text[-k:]
        if pattern.strip(_PUNCT_CHARS + " \t") == "":
            continue
        if all(
            text[-(r + 1) * k : -r * k] == pattern
            for r in range(1, pattern_repeat_thresh)
        ):
            return True, f"tail_pattern:'{pattern}'x{pattern_repeat_thresh}+"

    norm_tail = _PUNCT_RE.sub("", text).lower()
    n2 = len(norm_tail)
    for k in range(3, max_pattern_len + 1):
        if n2 < k * pattern_repeat_thresh:
            continue
        pattern = norm_tail[-k:]
        if all(
            norm_tail[-(r + 1) * k : -r * k] == pattern
            for r in range(1, pattern_repeat_thresh)
        ):
            return True, f"tail_pattern_norm:'{pattern}'x{pattern_repeat_thresh}+"
    return False, ""


def split_text_to_tokens(text: str) -> List[str]:
    """Split into English words and single non-English characters, no punctuation."""
    text = text.translate(_STRIP_PUNCT)
    return [m.group() for m in _TOKEN_RE.finditer(text) if m.group().strip()]


def is_chinese_token(token: str) -> bool:
    return any("一" <= ch <= "鿿" for ch in token)
