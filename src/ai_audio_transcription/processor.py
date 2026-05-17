from pathlib import Path

import numpy as np
from openai import OpenAI

from ai_audio_transcription.config import Settings
from ai_audio_transcription.logging_config import get_logger
from ai_audio_transcription.schemas import AudioProcessResult, TranscriptionUsage
from ai_audio_transcription.transcriber import OpenRouterTranscriber

log = get_logger("processor")

SYSTEM_PROMPT = (
    "Ты обрабатываешь транскрипцию речи по инструкции пользователя. "
    "Верни только итоговый текст без пояснений и метаданных."
)


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
        log.info("Обработка записи с микрофона (в памяти, без файла на диске)")
        stt = self._transcriber.transcribe_array(
            audio,
            sample_rate=sample_rate,
            model=stt_model,
            language=language,
            temperature=temperature,
        )
        return self._after_stt(stt, post_prompt=post_prompt, chat_model=chat_model)

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
            log.info("Постобработка LLM: модель=%s", model)
            text, chat_usage = self._apply_post_prompt(stt.text, post_prompt, model=model)
        else:
            log.info("LLM пропущена (нет промпта/перевода)")

        log.info("Обработка завершена")
        return AudioProcessResult(
            text=text,
            transcript=stt.text if post_prompt else None,
            stt_usage=stt.usage,
            chat_usage=chat_usage,
        )

    def _apply_post_prompt(
        self,
        transcript: str,
        post_prompt: str,
        *,
        model: str,
    ) -> tuple[str, TranscriptionUsage | None]:
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Инструкция:\n{post_prompt}\n\nТранскрипция:\n{transcript}",
                },
            ],
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
        log.info("LLM готова: %s символов", len(content))
        return content.strip(), usage
