# AI Audio Transcription (OpenRouter)

Speech-to-text через [OpenRouter STT API](https://openrouter.ai/docs/guides/overview/multimodal/stt) с **OpenAI Python SDK** (`base_url` → OpenRouter).

## Установка

```bash
poetry install
cp .env.example .env
# укажите OPENROUTER_API_KEY и модель
```

## Запуск

```bash
poetry run app
```

Два режима (переключатель вверху):

| Режим | Поведение |
|-------|-----------|
| **Пакетная запись** | Запись → Стоп → один ответ |
| **Лайв (по паузе)** | Слушает непрерывно; после паузы — STT + стрим перевода в две колонки |

Также: выбор аудиофайла с диска, промпт/перевод, настройка STT и LLM моделей.

Лайв: короткие промпты LLM (`LIVE_LLM_MAX_TOKENS=256`), порог паузы `LIVE_PAUSE_MS=800`.

## Модели (подсказка)

### STT (распознавание)

| Модель | Когда брать |
| --- | --- |
| `openai/whisper-1` | Дёшево, поминутная оплата |
| `openai/whisper-large-v3-turbo` | Баланс цена/скорость (дефолт) |
| `openai/whisper-large-v3` | Максимум качества |

### LLM (перевод / промпт)

По умолчанию: `google/gemini-2.0-flash-001`.

| Модель | Когда брать |
| --- | --- |
| `google/gemini-2.0-flash-001` | Дефолт для MVP |
| `meta-llama/llama-3.1-8b-instruct` | Ещё дешевле |
| `deepseek/deepseek-chat` | Низкая цена на OpenRouter |

Актуальные цены: [openrouter.ai/models](https://openrouter.ai/models).
