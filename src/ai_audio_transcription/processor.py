from collections.abc import Iterator
from pathlib import Path

import numpy as np
from openai import OpenAI

from ai_audio_transcription.config import Settings
from ai_audio_transcription.logging_config import get_logger
from ai_audio_transcription.prompts import (
    batch_llm_system,
    build_live_user_content,
    default_stt_prompt,
    live_llm_system,
)
from ai_audio_transcription.schemas import AudioProcessResult, TranscriptionUsage
from ai_audio_transcription.transcriber import OpenRouterTranscriber

log = get_logger("processor")


class AudioProcessor:
    def __init__(self, client: OpenAI, settings: Settings) -> None:
        self._client = client
        self._settings = settings
        self._transcriber = OpenRouterTranscriber(client)

    def process_file(
        self,
        audio_path: Path,
        *,
        stt_model: str,
        language: str | None = None,
        temperature: float | None = None,
        post_prompt: str | None = None,
        chat_model: str | None = None,
    ) -> AudioProcessResult:
        log.info("Обработка файла: %s", audio_path)
        stt = self._transcriber.transcribe(
            audio_path,
            model=stt_model,
            language=language,
            temperature=temperature,
            prompt=self._stt_prompt(),
        )
        return self._after_stt(stt, post_prompt=post_prompt, chat_model=chat_model)

    def process_audio(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        stt_model: str,
        language: str | None = None,
        temperature: float | None = None,
        post_prompt: str | None = None,
        chat_model: str | None = None,
    ) -> AudioProcessResult:
        log.info("Обработка аудио в памяти")
        stt = self._transcriber.transcribe_array(
            audio,
            sample_rate=sample_rate,
            model=stt_model,
            language=language,
            temperature=temperature,
            prompt=self._stt_prompt(),
        )
        return self._after_stt(stt, post_prompt=post_prompt, chat_model=chat_model)

    def transcribe_segment(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int,
        stt_model: str,
        language: str | None = None,
        temperature: float | None = None,
    ) -> str:
        log.info("Лайв STT сегмента")
        result = self._transcriber.transcribe_array(
            audio,
            sample_rate=sample_rate,
            model=stt_model,
            language=language,
            temperature=temperature,
            prompt=self._stt_prompt(),
        )
        return result.text

    def stream_live_post(
        self,
        transcript: str,
        post_prompt_template: str,
        *,
        model: str,
        max_tokens: int = 256,
    ) -> Iterator[str]:
        """Стрим перевода/постобработки для лайв-режима (короткий промпт)."""
        user_content = build_live_user_content(post_prompt_template, transcript)
        log.info("Лайв LLM stream: model=%s", model)
        stream = self._client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": live_llm_system(glossary=self._settings.openrouter_glossary),
                },
                {"role": "user", "content": user_content},
            ],
            extra_body={
                "session_id": "audio_stream_live_post",
            },
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _after_stt(
        self,
        stt,
        *,
        post_prompt: str | None,
        chat_model: str | None,
    ) -> AudioProcessResult:
        text = stt.text
        chat_usage: TranscriptionUsage | None = None

        if post_prompt:
            model = chat_model or self._settings.openrouter_chat_model
            log.info("Batch LLM: model=%s", model)
            text, chat_usage = self._apply_batch_post(stt.text, post_prompt, model=model)
        else:
            log.info("LLM пропущена (нет промпта/перевода)")

        return AudioProcessResult(
            text=text,
            transcript=stt.text if post_prompt else None,
            stt_usage=stt.usage,
            chat_usage=chat_usage,
        )

    def _apply_batch_post(
        self,
        transcript: str,
        post_prompt: str,
        *,
        model: str,
    ) -> tuple[str, TranscriptionUsage | None]:
        response = self._client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": batch_llm_system(glossary=self._settings.openrouter_glossary),
                },
                {
                    "role": "user",
                    "content": f"Инструкция:\n{post_prompt}\n\nТранскрипция:\n{transcript}",
                },
            ],
            extra_body={
                "session_id": "audio_batch_post",
            },
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM вернул пустой ответ.")

        usage: TranscriptionUsage | None = None
        if response.usage:
            usage = TranscriptionUsage(
                total_tokens=response.usage.total_tokens,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )
        return content.strip(), usage

    def _stt_prompt(self) -> str | None:
        if self._settings.openrouter_stt_prompt:
            return self._settings.openrouter_stt_prompt.strip() or None
        return default_stt_prompt(glossary=self._settings.openrouter_glossary)
