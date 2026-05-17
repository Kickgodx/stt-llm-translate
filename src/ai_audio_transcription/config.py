from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_audio_transcription.model_catalog import (
    DEFAULT_CHAT_MODEL,
    DEFAULT_STT_MODEL,
    resolve_chat_model,
    resolve_stt_model,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str
    openrouter_model: str = DEFAULT_STT_MODEL
    openrouter_chat_model: str = DEFAULT_CHAT_MODEL
    openrouter_language: str | None = None
    # Через запятую: python, sdet, automation — дополняют промпты STT/LLM
    openrouter_glossary: str | None = None
    # Подсказка Whisper (экспериментально; OpenRouter может игнорировать)
    openrouter_stt_prompt: str | None = None
    record_sample_rate: int = 16000
    record_ram_max_seconds: float = 300.0
    live_pause_ms: float = 800.0
    live_min_segment_ms: float = 400.0
    live_max_segment_ms: float = 30_000.0
    live_speech_rms_threshold: float = 400.0
    live_llm_max_tokens: int = 256
    log_level: str = "INFO"
    http_referer: str | None = None
    x_openrouter_title: str | None = None

    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    def resolved_models(self) -> tuple[str, str]:
        """STT и LLM с учётом .env и каталога."""
        return (
            resolve_stt_model(self.openrouter_model),
            resolve_chat_model(self.openrouter_chat_model),
        )
