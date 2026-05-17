import base64
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from ai_audio_transcription.logging_config import get_logger
from ai_audio_transcription.recorder import audio_to_wav_bytes
from ai_audio_transcription.schemas import (
    InputAudio,
    SttModelInfo,
    TranscriptionResult,
    TranscriptionUsage,
)

log = get_logger("stt")

SUPPORTED_EXTENSIONS = {
    ".wav": "wav",
    ".mp3": "mp3",
    ".flac": "flac",
    ".m4a": "m4a",
    ".ogg": "ogg",
    ".webm": "webm",
    ".aac": "aac",
}


def audio_format_from_path(path: Path) -> str:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Формат {ext!r} не поддерживается. Допустимо: {supported}")
    return SUPPORTED_EXTENSIONS[ext]


def encode_audio_file(path: Path) -> InputAudio:
    audio_format = audio_format_from_path(path)
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return InputAudio(data=data, format=audio_format)


def encode_audio_array(audio: np.ndarray, *, sample_rate: int) -> InputAudio:
    wav_bytes = audio_to_wav_bytes(audio, sample_rate=sample_rate)
    data = base64.b64encode(wav_bytes).decode("utf-8")
    return InputAudio(data=data, format="wav")


def _as_dict(response: object) -> dict[str, Any]:
    if not isinstance(response, dict):
        raise RuntimeError(f"Неожиданный формат ответа API: {type(response)!r}")
    return response


class OpenRouterTranscriber:
    """STT через OpenRouter: JSON + base64 (не multipart OpenAI)."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def transcribe(
        self,
        audio_path: Path,
        *,
        model: str,
        language: str | None = None,
        temperature: float | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        log.info("STT из файла: %s, модель=%s", audio_path.name, model)
        return self.transcribe_input(
            encode_audio_file(audio_path),
            model=model,
            language=language,
            temperature=temperature,
            prompt=prompt,
        )

    def transcribe_array(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        model: str,
        language: str | None = None,
        temperature: float | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        duration = len(audio.reshape(-1)) / sample_rate
        log.info("STT из памяти: %.2f с аудио, модель=%s", duration, model)
        return self.transcribe_input(
            encode_audio_array(audio, sample_rate=sample_rate),
            model=model,
            language=language,
            temperature=temperature,
            prompt=prompt,
        )

    def transcribe_input(
        self,
        input_audio: InputAudio,
        *,
        model: str,
        language: str | None = None,
        temperature: float | None = None,
        prompt: str | None = None,
    ) -> TranscriptionResult:
        body: dict[str, object] = {
            "model": model,
            "input_audio": input_audio.model_dump(),
        }
        if language:
            body["language"] = language
        if temperature is not None:
            body["temperature"] = temperature
        if prompt:
            body["prompt"] = prompt

        log.info("Отправка запроса STT…")
        response = _as_dict(
            self._client.post(
                "/audio/transcriptions",
                body=body,
                cast_to=object,
            )
        )

        usage_raw = response.get("usage")
        usage = TranscriptionUsage.model_validate(usage_raw) if usage_raw else None
        text = response.get("text")
        if not text:
            raise RuntimeError(f"Пустой ответ транскрипции: {response}")

        log.info("STT готово: %s символов", len(text))
        if usage and usage.cost is not None:
            log.info("STT cost=$%.6f", usage.cost)

        return TranscriptionResult(text=text, usage=usage, raw=response)

    def list_stt_models(self) -> list[SttModelInfo]:
        log.info("Загрузка списка STT-моделей…")
        response = _as_dict(
            self._client.get(
                "/models",
                options={"params": {"output_modalities": "transcription"}},
                cast_to=object,
            )
        )

        models: list[SttModelInfo] = []
        for item in response.get("data", []):
            modalities = item.get("output_modalities") or []
            if "transcription" not in modalities:
                continue
            models.append(
                SttModelInfo(
                    id=item["id"],
                    name=item.get("name"),
                    description=item.get("description"),
                    pricing=item.get("pricing"),
                )
            )

        log.info("Найдено STT-моделей: %s", len(models))
        return sorted(models, key=lambda m: m.id)
