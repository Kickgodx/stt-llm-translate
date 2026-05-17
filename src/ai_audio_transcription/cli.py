from pathlib import Path

import typer
from dotenv import load_dotenv

from ai_audio_transcription.client import create_openrouter_client
from ai_audio_transcription.config import Settings
from ai_audio_transcription.logging_config import setup_logging
from ai_audio_transcription.processor import AudioProcessor
from ai_audio_transcription.prompts import resolve_post_prompt
from ai_audio_transcription.recorder import record_to_wav
from ai_audio_transcription.schemas import AudioProcessResult, TranscriptionUsage
from ai_audio_transcription.transcriber import OpenRouterTranscriber

_PROCESS_OPTS = {
    "model": typer.Option(None, "--model", "-m", help="STT-модель OpenRouter"),
    "chat_model": typer.Option(None, "--chat-model", help="LLM для постобработки (--prompt)"),
    "language": typer.Option(
        None,
        "--language",
        "-l",
        help="ISO-639-1 для STT; иначе OPENROUTER_LANGUAGE",
    ),
    "temperature": typer.Option(
        None,
        "--temperature",
        "-t",
        min=0.0,
        max=1.0,
        help="Температура STT (0–1)",
    ),
    "prompt": typer.Option(
        None,
        "--prompt",
        "-p",
        help="Инструкция LLM после транскрипции (перевод, резюме и т.д.)",
    ),
    "translate": typer.Option(
        None,
        "--translate",
        help="Шорткат перевода: en | ru (вместо --prompt)",
    ),
    "output": typer.Option(None, "--output", "-o", help="Сохранить итоговый текст в файл"),
    "show_transcript": typer.Option(
        False,
        "--show-transcript",
        help="Показать сырую транскрипцию при использовании --prompt",
    ),
    "show_usage": typer.Option(True, "--usage/--no-usage", help="Показать usage/cost"),
}


def _load_settings() -> Settings:
    load_dotenv()
    settings = Settings()
    setup_logging(settings.log_level)
    return settings


def _build_processor(settings: Settings) -> AudioProcessor:
    client = create_openrouter_client(settings)
    return AudioProcessor(client, settings)


def _print_result(
    result: AudioProcessResult,
    *,
    output: Path | None,
    show_transcript: bool,
    show_usage: bool,
) -> None:
    if show_transcript and result.transcript:
        typer.echo("--- Транскрипция ---", err=True)
        typer.echo(result.transcript, err=True)
        typer.echo("--- Результат ---", err=True)

    typer.echo(result.text)
    if output:
        output.write_text(result.text, encoding="utf-8")
        typer.echo(f"\nСохранено: {output}", err=True)

    if not show_usage:
        return

    parts: list[str] = []
    if result.stt_usage:
        parts.extend(_usage_parts(result.stt_usage, prefix="stt"))
    if result.chat_usage:
        parts.extend(_usage_parts(result.chat_usage, prefix="llm"))
    if parts:
        typer.echo(" | ".join(parts), err=True)


def _usage_parts(usage: TranscriptionUsage, *, prefix: str) -> list[str]:
    parts: list[str] = []
    if usage.seconds is not None:
        parts.append(f"{prefix}_seconds={usage.seconds:.2f}")
    if usage.cost is not None:
        parts.append(f"{prefix}_cost=${usage.cost:.6f}")
    if usage.total_tokens is not None:
        parts.append(f"{prefix}_tokens={usage.total_tokens}")
    return parts


def _process(
    processor: AudioProcessor,
    audio_path: Path,
    settings: Settings,
    *,
    model: str | None,
    chat_model: str | None,
    language: str | None,
    temperature: float | None,
    prompt: str | None,
    translate: str | None,
    output: Path | None,
    show_transcript: bool,
    show_usage: bool,
) -> None:
    try:
        post_prompt = resolve_post_prompt(prompt=prompt, translate=translate)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc
    result = processor.process_file(
        audio_path,
        stt_model=model or settings.openrouter_model,
        language=language or settings.openrouter_language,
        temperature=temperature,
        post_prompt=post_prompt,
        chat_model=chat_model,
    )
    _print_result(
        result,
        output=output,
        show_transcript=show_transcript,
        show_usage=show_usage,
    )


def transcribe(
    audio: Path = typer.Argument(..., exists=True, dir_okay=False, help="Путь к аудиофайлу"),
    model: str | None = _PROCESS_OPTS["model"],
    chat_model: str | None = _PROCESS_OPTS["chat_model"],
    language: str | None = _PROCESS_OPTS["language"],
    temperature: float | None = _PROCESS_OPTS["temperature"],
    prompt: str | None = _PROCESS_OPTS["prompt"],
    translate: str | None = _PROCESS_OPTS["translate"],
    output: Path | None = _PROCESS_OPTS["output"],
    show_transcript: bool = _PROCESS_OPTS["show_transcript"],
    show_usage: bool = _PROCESS_OPTS["show_usage"],
) -> None:
    """Транскрибировать аудиофайл (опционально — LLM по --prompt)."""
    settings = _load_settings()
    _process(
        _build_processor(settings),
        audio,
        settings,
        model=model,
        chat_model=chat_model,
        language=language,
        temperature=temperature,
        prompt=prompt,
        translate=translate,
        output=output,
        show_transcript=show_transcript,
        show_usage=show_usage,
    )


def record(
    duration: float | None = typer.Option(
        None,
        "--duration",
        "-d",
        min=0.1,
        help="Секунды записи; без флага — Enter для остановки",
    ),
    device: int | None = typer.Option(None, "--device", help="ID устройства sounddevice"),
    model: str | None = _PROCESS_OPTS["model"],
    chat_model: str | None = _PROCESS_OPTS["chat_model"],
    language: str | None = _PROCESS_OPTS["language"],
    temperature: float | None = _PROCESS_OPTS["temperature"],
    prompt: str | None = _PROCESS_OPTS["prompt"],
    translate: str | None = _PROCESS_OPTS["translate"],
    output: Path | None = _PROCESS_OPTS["output"],
    show_transcript: bool = _PROCESS_OPTS["show_transcript"],
    show_usage: bool = _PROCESS_OPTS["show_usage"],
    keep_audio: Path | None = typer.Option(
        None,
        "--keep-audio",
        help="Сохранить WAV записи (путь к файлу)",
    ),
) -> None:
    """Записать с микрофона, STT, опционально LLM, вывод текста."""
    settings = _load_settings()

    if duration is None:
        typer.echo("Запись с микрофона. Нажмите Enter, чтобы остановить.", err=True)
    else:
        typer.echo(f"Запись {duration:.1f} с...", err=True)

    wav_path = record_to_wav(
        sample_rate=settings.record_sample_rate,
        duration=duration,
        device=device,
    )

    if keep_audio:
        keep_audio.write_bytes(wav_path.read_bytes())
        typer.echo(f"Аудио: {keep_audio}", err=True)

    try:
        _process(
            _build_processor(settings),
            wav_path,
            settings,
            model=model,
            chat_model=chat_model,
            language=language,
            temperature=temperature,
            prompt=prompt,
            translate=translate,
            output=output,
            show_transcript=show_transcript,
            show_usage=show_usage,
        )
    finally:
        wav_path.unlink(missing_ok=True)


def list_stt_models() -> None:
    """Вывести STT-модели OpenRouter (output_modalities=transcription)."""
    settings = _load_settings()
    client = create_openrouter_client(settings)
    transcriber = OpenRouterTranscriber(client)
    models = transcriber.list_stt_models()

    if not models:
        typer.echo("STT-модели не найдены.", err=True)
        raise typer.Exit(1)

    for m in models:
        price = ""
        if m.pricing:
            p = m.pricing.get("prompt")
            c = m.pricing.get("completion")
            if p is not None or c is not None:
                price = f"  prompt={p} completion={c}"
        typer.echo(f"{m.id}{price}")
        if m.name:
            typer.echo(f"  {m.name}")
        if m.description:
            typer.echo(f"  {m.description.split(chr(10), 1)[0][:120]}")


def run_transcribe() -> None:
    typer.run(transcribe)


def run_record() -> None:
    typer.run(record)
