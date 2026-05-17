# AI Audio Transcription (OpenRouter)

Speech-to-text через [OpenRouter STT API](https://openrouter.ai/docs/guides/overview/multimodal/stt) с **OpenAI Python SDK** (`base_url` → OpenRouter).

## Установка

```bash
poetry install
cp .env.example .env
# укажите OPENROUTER_API_KEY и модель
```

## Использование

### GUI (рекомендуется)

```bash
poetry run app
```

Кнопка **Запись** — старт; **Стоп и отправить** — остановка, STT и (при промпте/переводе) LLM. **Гибрид:** до 5 мин (`RECORD_RAM_MAX_SECONDS=300`) — буфер в RAM, дольше — автоматически temp-WAV; после отправки temp **удаляется**. Логи в консоли (`LOG_LEVEL=INFO`).

### CLI

```bash
# транскрипция файла
poetry run transcribe path/to/audio.mp3

# запись с микрофона (Enter — остановить)
poetry run record

# запись 5 секунд + перевод на английский
poetry run record --duration 5 --translate en --language ru

# свой промпт после транскрипции
poetry run record --prompt "Переведи на английский. Только перевод."

# с языком и другой моделью
poetry run transcribe audio.wav --language ru --model openai/whisper-1

# список STT-моделей на OpenRouter
poetry run list-stt-models
```

Пайплайн с `--prompt` / `--translate`: **микрофон/файл → Whisper (STT) → LLM** (`OPENROUTER_CHAT_MODEL`) по вашей инструкции.

## Модели (подсказка)

### STT (распознавание)

| Модель | Когда брать |
| --- | --- |
| `openai/whisper-1` | Дёшево, поминутная оплата, проверенная классика |
| `openai/whisper-large-v3-turbo` | Баланс цена/скорость, много языков |
| `openai/whisper-large-v3` | Максимум качества, шумные записи |

### LLM (перевод / промпт, только при `--prompt` или переводе)

По умолчанию: `google/gemini-2.0-flash-001` — дёшево для MVP.

| Модель | Когда брать |
| --- | --- |
| `google/gemini-2.0-flash-001` | Дефолт: перевод, простые инструкции |
| `meta-llama/llama-3.1-8b-instruct` | Ещё дешевле, базовые задачи |
| `deepseek/deepseek-chat` | Очень низкая цена на OpenRouter |

Актуальные цены: [openrouter.ai/models](https://openrouter.ai/models).
