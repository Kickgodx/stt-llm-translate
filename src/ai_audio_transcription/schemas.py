from typing import Any

from pydantic import BaseModel, Field


class InputAudio(BaseModel):
    data: str
    format: str


class TranscriptionUsage(BaseModel):
    seconds: float | None = None
    total_tokens: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None


class TranscriptionResult(BaseModel):
    text: str
    usage: TranscriptionUsage | None = None
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)


class SttModelInfo(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    pricing: dict[str, Any] | None = None


class AudioProcessResult(BaseModel):
    text: str
    transcript: str | None = None
    stt_usage: TranscriptionUsage | None = None
    chat_usage: TranscriptionUsage | None = None
