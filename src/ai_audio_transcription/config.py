from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str
    openrouter_model: str = "openai/whisper-large-v3-turbo"
    openrouter_chat_model: str = "google/gemini-2.0-flash-001"
    openrouter_language: str | None = None
    record_sample_rate: int = 16000
    record_ram_max_seconds: float = 300.0
    log_level: str = "INFO"
    http_referer: str | None = None
    x_openrouter_title: str | None = None

    openrouter_base_url: str = "https://openrouter.ai/api/v1"
