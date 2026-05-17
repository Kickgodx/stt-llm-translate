from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from ai_audio_transcription.client import create_openrouter_client
from ai_audio_transcription.config import Settings
from ai_audio_transcription.logging_config import get_logger, setup_logging
from ai_audio_transcription.processor import AudioProcessor
from ai_audio_transcription.prompts import resolve_post_prompt
from ai_audio_transcription.schemas import AudioProcessResult, TranscriptionUsage

log = get_logger("service")


@dataclass
class ProcessOptions:
    model: str | None = None
    chat_model: str | None = None
    language: str | None = None
    temperature: float | None = None
    prompt: str | None = None
    translate: str | None = None


def format_result(
    result: AudioProcessResult,
    *,
    show_transcript: bool,
    show_usage: bool,
) -> str:
    parts: list[str] = []
    if show_transcript and result.transcript:
        parts.extend(["--- Транскрипция ---", result.transcript, "--- Результат ---"])
    parts.append(result.text)

    if show_usage:
        usage_lines = _format_usage_lines(result)
        if usage_lines:
            parts.append("")
            parts.extend(usage_lines)
    return "\n".join(parts)


def _format_usage_lines(result: AudioProcessResult) -> list[str]:
    lines: list[str] = []
    if result.stt_usage:
        lines.append(_usage_line("STT", result.stt_usage))
    if result.chat_usage:
        lines.append(_usage_line("LLM", result.chat_usage))
    return lines


def _usage_line(label: str, usage: TranscriptionUsage) -> str:
    bits: list[str] = []
    if usage.seconds is not None:
        bits.append(f"{usage.seconds:.1f} с")
    if usage.total_tokens is not None:
        bits.append(f"{usage.total_tokens} tok")
    if usage.cost is not None:
        bits.append(f"${usage.cost:.6f}")
    return f"{label}: {', '.join(bits)}" if bits else ""


class AudioService:
    def __init__(self) -> None:
        load_dotenv()
        self.settings = Settings()
        setup_logging(self.settings.log_level)
        log.info(
            "Сервис запущен: STT=%s, LLM=%s",
            self.settings.openrouter_model,
            self.settings.openrouter_chat_model,
        )
        client = create_openrouter_client(self.settings)
        self.processor = AudioProcessor(client, self.settings)

    def _resolved_options(self, options: ProcessOptions) -> tuple[ProcessOptions, str | None]:
        post_prompt = resolve_post_prompt(prompt=options.prompt, translate=options.translate)
        if post_prompt:
            log.info("Включена постобработка (промпт/перевод)")
        return options, post_prompt

    def process_file(self, audio_path: Path, options: ProcessOptions) -> AudioProcessResult:
        options, post_prompt = self._resolved_options(options)
        return self.processor.process_file(
            audio_path,
            stt_model=options.model or self.settings.openrouter_model,
            language=options.language or self.settings.openrouter_language,
            temperature=options.temperature,
            post_prompt=post_prompt,
            chat_model=options.chat_model,
        )

    def process_audio(
        self,
        audio: np.ndarray,
        options: ProcessOptions,
    ) -> AudioProcessResult:
        options, post_prompt = self._resolved_options(options)
        return self.processor.process_audio(
            audio,
            sample_rate=self.settings.record_sample_rate,
            stt_model=options.model or self.settings.openrouter_model,
            language=options.language or self.settings.openrouter_language,
            temperature=options.temperature,
            post_prompt=post_prompt,
            chat_model=options.chat_model,
        )
