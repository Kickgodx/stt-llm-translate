# AI Audio Transcription (OpenRouter)

Speech-to-text через [OpenRouter STT API](https://openrouter.ai/docs/guides/overview/multimodal/stt) с **OpenAI Python SDK** (`base_url` → OpenRouter).

## Установка

```bash
poetry install
cp .env.example .env
# укажите OPENROUTER_API_KEY
```

## Запуск

```bash
poetry run app
```

Два режима: **Пакетная запись** и **Лайв (по паузе)**. Модели STT и LLM — выпадающие списки в UI.

## Каталог моделей (одна точка входа)

Список моделей задаётся в **`src/ai_audio_transcription/model_catalog.py`**:

- `STT_MODELS` — для распознавания речи
- `CHAT_MODELS` — для перевода / промпта

Дефолты `DEFAULT_STT_MODEL` / `DEFAULT_CHAT_MODEL` и поля в `config.py` ссылаются на этот файл.

В `.env` можно переопределить выбранную по умолчанию модель:

```env
OPENROUTER_MODEL=openai/whisper-1
OPENROUTER_CHAT_MODEL=google/gemini-2.0-flash-001
```

Если id из `.env` нет в каталоге, он всё равно появится в выпадающем списке (пометка «из .env»).

## Прочие настройки `.env`

См. `.env.example` — `LIVE_PAUSE_MS`, `LOG_LEVEL`, и т.д.
