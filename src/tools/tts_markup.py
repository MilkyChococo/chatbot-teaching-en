import json
import os
import re
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from src.utils.load_model import load_chat_model
except ImportError:  # pragma: no cover - fallback for graph/api runtime
    from utils.load_model import load_chat_model


TTS_EN_VOICE = os.getenv("GCP_TTS_VOICE_EN", "en-US-Neural2-C")
_JSON_RE = re.compile(r"\{.*\}", re.S)
_VI_DIACRITIC_RE = re.compile(r"[À-ỹà-ỹĐđ]")
_EN_WORD_RE = re.compile(r"[A-Za-z]{2,}")
_INLINE_EN_PHRASE_RE = re.compile(
    r"(?<![^\W\d_])[A-Za-z][A-Za-z'-]*(?:\s+[A-Za-z][A-Za-z'-]*)*(?![^\W\d_])"
)
_VOICE_SEGMENT_RE = re.compile(r"<voice\b[^>]*>(.*?)</voice>", re.S)
_MERGEABLE_VOICE_RE = re.compile(
    r'<voice name="([^"]+)">(.*?)</voice>([\s,.;:!?()\[\]\'"“”‘’/-]+)<voice name="\1">(.*?)</voice>',
    re.S,
)
_TRAILING_VOICE_PUNCT_RE = re.compile(
    r'<voice name="([^"]+)">(.*?)</voice>([,.!?;:]+)',
    re.S,
)
_ASCII_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_SHORT_ENGLISH_ALLOWLIST = {
    "ai",
    "api",
    "app",
    "bot",
    "chat",
    "city",
    "code",
    "cv",
    "data",
    "game",
    "goal",
    "hr",
    "idea",
    "it",
    "job",
    "mock",
    "plan",
    "qa",
    "role",
    "sql",
    "test",
    "topic",
    "tour",
    "trip",
    "ui",
    "ux",
    "work",
}
_ENGLISH_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "do",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "my",
    "no",
    "of",
    "on",
    "or",
    "our",
    "that",
    "the",
    "their",
    "this",
    "to",
    "we",
    "with",
    "you",
    "your",
    "yes",
}
_NO_VOICE_PHRASES = {
    "anh",
    "du",
    "du lich",
    "du lịch",
    "giai tri",
    "giải trí",
    "giao",
    "giao tiep",
    "giao tiep hang ngay",
    "giao tiếp",
    "giao tiếp hàng ngày",
    "gia",
    "gia su",
    "gia sư",
    "sau",
    "ta",
    "tieng anh",
    "tiếng anh",
    "trung bình",
    "trung binh"
}

_TTS_MARKUP_PROMPT = f"""
You convert a tutoring assistant reply into Google Cloud Text-to-Speech SSML.

Return strict JSON only:
{{"assistant_tts_ssml":"<speak>...</speak>"}}

Rules:
- Preserve the original wording exactly. Do not translate, paraphrase, reorder, add, or remove words.
- Wrap the full output in <speak>...</speak>.
- Keep Vietnamese text as plain text. Some words like "gia sư", "du lịch", "công việc" are Vietnamese and should be outside English voice tags.
- Vietnamese text MUST stay outside English voice tags.
- Wrap every English word or English phrase in <voice name="{TTS_EN_VOICE}">...</voice>.
- Merge adjacent English words into a single <voice> tag when they belong to one phrase.
- If the whole sentence is English, wrap the full sentence in one <voice> tag.
- Escape XML reserved characters inside text nodes: &, <, >, ", '.
- Do not use markdown, code fences, comments, or explanations.

Example 1
Input: Hôm nay mình sẽ mock interview để luyện coding nhé.
Output:
{{"assistant_tts_ssml":"<speak>Hôm nay mình sẽ <voice name=\\"{TTS_EN_VOICE}\\">mock interview</voice> để luyện <voice name=\\"{TTS_EN_VOICE}\\">coding</voice> nhé.</speak>"}}

Example 2
Input: Bạn hãy nói lại câu This is my favorite book.
Output:
{{"assistant_tts_ssml":"<speak>Bạn hãy nói lại câu <voice name=\\"{TTS_EN_VOICE}\\">This is my favorite book.</voice></speak>"}}

Example 3
Input: Chào bạn! Tôi là gia sư tiếng Anh của bạn. Chúng ta hãy cùng luyện tập nhé. Bạn muốn chọn chủ đề nào sau đây? 1. Giao tiếp hàng ngày 2. Du lịch 3. Công việc 4. Học tập 5. Sức khỏe 6. Giải trí
Output:
{{"assistant_tts_ssml":"<speak>Chào bạn! Tôi là gia sư tiếng Anh của bạn. Chúng ta hãy cùng luyện tập nhé. Bạn muốn chọn chủ đề nào sau đây? 1. Giao tiếp hàng ngày 2. Du lịch 3. Công việc 4. Học tập 5. Sức khỏe 6. Giải trí</speak>"}}

Example 4
Input: Xin chào bạn, đây là chatbot hỗ trợ học tiếng Anh. Bạn vui lòng nói tiếng Anh trong suốt quá trình nhé.
Output:
{{"assistant_tts_ssml":"<speak>Xin chào bạn, đây là <voice name=\\"{TTS_EN_VOICE}\\">chatbot</voice> hỗ trợ học tiếng Anh. Bạn vui lòng nói tiếng Anh trong suốt quá trình nhé.</speak>"}}

Example 5
Input: Chào bạn! Bạn muốn chọn chủ đề Du lịch. Trình độ tiếng Anh của bạn hiện tại là: 1. Người mới bắt đầu 2. Trung bình 3. Tốt
Output:
{{"assistant_tts_ssml":"<speak>Chào bạn! Bạn muốn chọn chủ đề <voice name=\\"{TTS_EN_VOICE}\\">Du lịch</voice>. Trình độ tiếng Anh của bạn hiện tại là: 1. Người mới bắt đầu 2. Trung bình 3. Tốt</speak>"}}

Example 6
Input: Bạn đã nghe được 'no, I can't', nhưng câu tiếng Anh bắt đầu bằng một từ khác. Hãy thử nghe lại và tập trung vào những từ đầu tiên của câu nhé. Bạn thử lại xem sao? Luyện nghe – bước 1. Lên kế hoạch cho một chuyến đi đến một thành phố mới luôn thú vị! English: Planning a trip to a new city is always exciting! Bạn nhắc lại câu tiếng Anh này nhé.
Output:
{{"assistant_tts_ssml":"<speak>Bạn đã nghe được '<voice name=\\"{TTS_EN_VOICE}\\">no, I can't</voice>', nhưng câu tiếng Anh bắt đầu bằng một từ khác. Hãy thử nghe lại và tập trung vào những từ đầu tiên của câu nhé. Bạn thử lại xem sao? Luyện nghe – bước 1. Lên kế hoạch cho một chuyến đi đến một thành phố mới luôn thú vị! <voice name=\\"{TTS_EN_VOICE}\\">Planning a trip to a new city is always exciting!</voice> Bạn nhắc lại câu tiếng Anh này nhé.</speak>"}}

Example 7
Input: Tốt lắm. Bạn đọc gần đúng. Luyện nghe – bước 3. Lên kế hoạch cho một chuyến đi có thể rất thú vị, từ việc chọn điểm đến đến việc đóng gói vali của bạn. English: Planning a trip can be exciting, from choosing a destination to packing your suitcase. Bạn nhắc lại câu tiếng Anh này nhé.
Output: {{"assistant_tts_ssml":"<speak>Tốt lắm. Bạn đọc gần đúng. Luyện nghe – bước 3. Lên kế hoạch cho một chuyến đi có thể rất thú vị, từ việc chọn điểm đến đến việc đóng gói vali của bạn. <voice name=\\"{TTS_EN_VOICE}\\">English: Planning a trip can be exciting, from choosing a destination to packing your suitcase.</voice> Bạn nhắc lại câu tiếng Anh này nhé.</speak>"}}
""".strip()


def _escape_ssml_text(value: str) -> str:
    return (
        (value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _basic_ssml(text: str) -> str:
    return f"<speak>{_escape_ssml_text(text)}</speak>"


def _english_only_ssml(text: str) -> str:
    escaped = _escape_ssml_text(text)
    return f'<speak><voice name="{TTS_EN_VOICE}">{escaped}</voice></speak>'


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text") or item.get("output_text") or item.get("content")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _extract_json_blob(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}

    try:
        return json.loads(text)
    except Exception:
        pass

    match = _JSON_RE.search(text)
    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except Exception:
        return {}


def _looks_mixed_vi_en(text: str) -> bool:
    return bool(_VI_DIACRITIC_RE.search(text or "")) and bool(_EN_WORD_RE.search(text or ""))


def _looks_english_only(text: str) -> bool:
    value = (text or "").strip()
    if not value:
        return False
    if _VI_DIACRITIC_RE.search(value):
        return False
    return len(_EN_WORD_RE.findall(value)) >= 3


def _is_valid_ssml(ssml: str) -> bool:
    value = (ssml or "").strip()
    return value.startswith("<speak>") and value.endswith("</speak>")


def _unescape_ssml_text(value: str) -> str:
    return (
        (value or "")
        .replace("&apos;", "'")
        .replace("&quot;", '"')
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&amp;", "&")
    )


def _strip_ssml_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value or "")
    return _unescape_ssml_text(text)


def _normalize_compare_text(value: str) -> str:
    return " ".join((value or "").split()).strip()


def _normalize_phrase_key(value: str) -> str:
    text = _normalize_compare_text(_unescape_ssml_text(value)).lower()
    text = re.sub(r"^[^\wÀ-ỹà-ỹ]+|[^\wÀ-ỹà-ỹ]+$", "", text)
    return text


def _ascii_tokens(value: str) -> list[str]:
    return _ASCII_TOKEN_RE.findall(value or "")


def _is_english_like_phrase(value: str) -> bool:
    if _normalize_phrase_key(value) in _NO_VOICE_PHRASES:
        return False

    text = _normalize_compare_text(_unescape_ssml_text(value))
    tokens = _ascii_tokens(text)
    if not tokens:
        return False

    lower_tokens = [token.lower() for token in tokens]
    if any(token.isupper() and 2 <= len(token) <= 5 for token in tokens):
        return True

    if len(tokens) == 1:
        token = lower_tokens[0]
        return len(token) >= 5 or token in _SHORT_ENGLISH_ALLOWLIST

    if any(token in _SHORT_ENGLISH_ALLOWLIST for token in lower_tokens):
        return True

    if any(len(token) >= 5 for token in lower_tokens):
        return True

    if any(token in _ENGLISH_FUNCTION_WORDS for token in lower_tokens) and any(
        len(token) >= 4 for token in lower_tokens
    ):
        return True

    return False


def _detect_lang(text: str) -> str:
    if _VI_DIACRITIC_RE.search(text or ""):
        return "vi-VN"
    if len(_EN_WORD_RE.findall(text or "")) >= 4:
        return "en-US"
    return "vi-VN"


def _split_quoted_segments(text: str, forced_lang: str | None = None) -> list[dict[str, str]]:
    value = text or ""
    segments: list[dict[str, str]] = []
    last = 0

    for match in re.finditer(r"(['\"])(.*?)\1", value):
        before = value[last:match.start()]
        if before.strip():
            segments.append({"text": before, "lang": forced_lang or _detect_lang(before)})

        inner = match.group(2)
        if inner.strip():
            segments.append({"text": inner, "lang": "en-US"})

        last = match.end()

    tail = value[last:]
    if tail.strip():
        segments.append({"text": tail, "lang": forced_lang or _detect_lang(tail)})

    return segments


def _split_bilingual_segments(text: str) -> list[dict[str, str]]:
    value = text or ""
    chunks = re.findall(r"[^.!?]+[.!?]*", value) or [value]
    segments: list[dict[str, str]] = []

    for chunk in chunks:
        sentence = chunk.strip()
        if not sentence:
            continue

        colon_index = sentence.find(":")
        if colon_index != -1:
            head = sentence[: colon_index + 1]
            tail = sentence[colon_index + 1 :]
            if tail.strip() and _detect_lang(tail) == "en-US":
                segments.extend(_split_quoted_segments(head))
                segments.extend(_split_quoted_segments(tail, "en-US"))
                continue

        last = 0
        for match in re.finditer(r"\([^)]*\)", sentence):
            before = sentence[last:match.start()]
            if before.strip():
                segments.extend(_split_quoted_segments(before))

            inner = match.group(0)[1:-1]
            if inner.strip():
                segments.append({"text": inner, "lang": "en-US"})
            last = match.end()

        tail = sentence[last:]
        if tail.strip():
            segments.extend(_split_quoted_segments(tail))

    if not segments and value.strip():
        segments.append({"text": value, "lang": _detect_lang(value)})

    return segments


def _build_rule_based_ssml(text: str) -> str:
    clean_text = text or ""
    if not clean_text.strip():
        return _basic_ssml(clean_text)

    if _looks_english_only(clean_text):
        return _english_only_ssml(clean_text)

    matches = list(_INLINE_EN_PHRASE_RE.finditer(clean_text))
    if not matches:
        return _basic_ssml(clean_text)

    body: list[str] = []
    last = 0
    for match in matches:
        start, end = match.span()
        phrase = match.group(0)
        if not phrase.strip():
            continue
        if not _is_english_like_phrase(phrase):
            continue
        if start > last:
            body.append(_escape_ssml_text(clean_text[last:start]))
        body.append(f'<voice name="{TTS_EN_VOICE}">{_escape_ssml_text(phrase)}</voice>')
        last = end

    if last < len(clean_text):
        body.append(_escape_ssml_text(clean_text[last:]))

    return _merge_adjacent_voice_segments(f"<speak>{''.join(body)}</speak>")


def _sanitize_model_ssml(ssml: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1) or ""
        plain_inner = _unescape_ssml_text(inner)
        if _VI_DIACRITIC_RE.search(plain_inner) or not _is_english_like_phrase(plain_inner):
            return inner
        return match.group(0)

    return _merge_adjacent_voice_segments(_VOICE_SEGMENT_RE.sub(_replace, ssml or ""))


def _merge_adjacent_voice_segments(ssml: str) -> str:
    merged = ssml or ""
    while True:
        next_merged = _MERGEABLE_VOICE_RE.sub(
            lambda match: (
                f'<voice name="{match.group(1)}">'
                f"{match.group(2)}{match.group(3)}{match.group(4)}"
                "</voice>"
            ),
            merged,
        )
        next_merged = _TRAILING_VOICE_PUNCT_RE.sub(
            lambda match: (
                f'<voice name="{match.group(1)}">'
                f"{match.group(2)}{match.group(3)}"
                "</voice>"
            ),
            next_merged,
        )
        if next_merged == merged:
            return merged
        merged = next_merged


def _has_invalid_voice_segments(ssml: str) -> bool:
    for match in _VOICE_SEGMENT_RE.finditer(ssml or ""):
        plain_inner = _unescape_ssml_text(match.group(1) or "")
        if _VI_DIACRITIC_RE.search(plain_inner):
            return True
        if not _is_english_like_phrase(plain_inner):
            return True
    return False


def _is_valid_mixed_language_ssml(original_text: str, ssml: str) -> bool:
    if not _is_valid_ssml(ssml):
        return False
    if _normalize_compare_text(_strip_ssml_tags(ssml)) != _normalize_compare_text(original_text):
        return False
    if not _looks_mixed_vi_en(original_text):
        return True
    if "<voice" not in ssml:
        return False
    if _has_invalid_voice_segments(ssml):
        return False
    return True


@lru_cache(maxsize=1)
def get_tts_markup_model():
    return load_chat_model(
        "google_vertexai/gemini-2.5-flash",
        tags=["tts_markup"],
        temperature=0.0,
    )


def generate_assistant_tts_ssml(text: str) -> str:
    clean_text = (text or "").strip()
    if not clean_text:
        return ""

    if _looks_english_only(clean_text):
        return _english_only_ssml(clean_text)

    if not _looks_mixed_vi_en(clean_text):
        return _basic_ssml(clean_text)

    model = get_tts_markup_model()
    response = model.invoke(
        [
            SystemMessage(content=_TTS_MARKUP_PROMPT),
            HumanMessage(content=f"Input text:\n{clean_text}"),
        ]
    )
    payload = _extract_json_blob(_response_text(getattr(response, "content", response)))
    ssml = _sanitize_model_ssml(str(payload.get("assistant_tts_ssml") or "").strip())
    if _is_valid_mixed_language_ssml(clean_text, ssml):
        return ssml

    return _build_rule_based_ssml(clean_text)
