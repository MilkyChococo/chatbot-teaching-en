import os
from functools import lru_cache

try:
    from google.cloud import texttospeech
except ImportError:  # pragma: no cover - handled at runtime in API/tests
    texttospeech = None


_DEFAULT_VOICES = {
    "en-US": ("en-US", os.getenv("GCP_TTS_VOICE_EN", "en-US-Neural2-C")),
    "vi-VN": ("vi-VN", os.getenv("GCP_TTS_VOICE_VI", "vi-VN-Neural2-A")),
}


def normalize_tts_lang(lang: str | None) -> str:
    value = (lang or "").strip().lower()
    if value.startswith("en"):
        return "en-US"
    return "vi-VN"


@lru_cache(maxsize=1)
def get_tts_client():
    if texttospeech is None:
        raise RuntimeError("google-cloud-texttospeech is not installed")
    return texttospeech.TextToSpeechClient()


def synthesize_speech_audio(
    text: str = "",
    ssml: str | None = None,
    lang: str = "vi-VN",
    speaking_rate: float = 1.0,
) -> bytes:
    clean_text = (text or "").strip()
    clean_ssml = (ssml or "").strip()
    if not clean_text and not clean_ssml:
        raise ValueError("text is empty")

    normalized_lang = normalize_tts_lang(lang)
    language_code, voice_name = _DEFAULT_VOICES[normalized_lang]
    client = get_tts_client()
    synthesis_input = (
        texttospeech.SynthesisInput(ssml=clean_ssml)
        if clean_ssml
        else texttospeech.SynthesisInput(text=clean_text)
    )

    response = client.synthesize_speech(
        input=synthesis_input,
        voice=texttospeech.VoiceSelectionParams(
            language_code=language_code,
            name=voice_name,
        ),
        audio_config=texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
        ),
    )

    return bytes(response.audio_content)
