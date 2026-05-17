from openai import OpenAI

from ai_audio_transcription.config import Settings


def create_openrouter_client(settings: Settings) -> OpenAI:
    extra_headers: dict[str, str] = {}
    if settings.http_referer:
        extra_headers["HTTP-Referer"] = settings.http_referer
    if settings.x_openrouter_title:
        extra_headers["X-OpenRouter-Title"] = settings.x_openrouter_title

    return OpenAI(
        base_url=settings.openrouter_base_url,
        api_key=settings.openrouter_api_key,
        default_headers=extra_headers or None,
    )
