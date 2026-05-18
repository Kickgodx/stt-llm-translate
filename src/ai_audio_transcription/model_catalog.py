"""Единый каталог моделей OpenRouter для STT и LLM.

Редактируйте списки здесь — UI, config и дефолты берут значения отсюда.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelOption:
    id: str
    label: str


# --- STT (output_modalities=transcription) ---
STT_MODELS: tuple[ModelOption, ...] = (
    ModelOption("openai/whisper-large-v3-turbo", "OpenAI Whisper Large v3 Turbo, 0.04/h"),
    ModelOption("openai/whisper-large-v3", "OpenAI Whisper Large v3, 0.111/h"),
    ModelOption("openai/whisper-1", "OpenAI Whisper 1, 0.36/h"),
)

# --- LLM (перевод / промпт) ---
CHAT_MODELS: tuple[ModelOption, ...] = (
    ModelOption("mistralai/mistral-small-24b-instruct-2501", "Mistral Small 3, 0.05/0.08"),
    ModelOption("mistralai/mistral-small-3.2-24b-instruct", "Mistral Small 3.2, 0.075/0.2"),
    ModelOption("mistralai/mistral-small-2603", "Mistral Small 4, 0.15/0.6"),
    ModelOption("google/gemini-2.0-flash-001", "Google Gemini 2.0 Flash, 0.1/0.4"),
    ModelOption("google/gemini-2.0-flash-lite-001", "Google Gemini 2.0 Flash Lite, 0.075/0.3"),
    ModelOption("meta-llama/llama-3.3-70b-instruct", "Meta Llama 3.3 70B Instruct, 0.1/0.32"),
    ModelOption("deepseek/deepseek-v3.2", "DeepSeek V3.2, 0.25/0.4"),
    ModelOption("openai/gpt-4o-mini", "OpenAI GPT-4o Mini, 0.15/0.6"),
)

DEFAULT_STT_MODEL: str = STT_MODELS[0].id
DEFAULT_CHAT_MODEL: str = CHAT_MODELS[0].id


def stt_model_ids() -> list[str]:
    return [m.id for m in STT_MODELS]


def chat_model_ids() -> list[str]:
    return [m.id for m in CHAT_MODELS]


def stt_dropdown_labels() -> list[str]:
    return [m.label for m in STT_MODELS]


def chat_dropdown_labels() -> list[str]:
    return [m.label for m in CHAT_MODELS]


def _index_by_id(options: tuple[ModelOption, ...]) -> dict[str, ModelOption]:
    return {m.id: m for m in options}


_STT_BY_ID = _index_by_id(STT_MODELS)
_CHAT_BY_ID = _index_by_id(CHAT_MODELS)


def resolve_stt_model(model_id: str | None) -> str:
    """Модель из .env или дефолт; неизвестный id добавляется в список для UI."""
    if not model_id:
        return DEFAULT_STT_MODEL
    return model_id


def resolve_chat_model(model_id: str | None) -> str:
    if not model_id:
        return DEFAULT_CHAT_MODEL
    return model_id


def stt_options_for_ui(selected_id: str) -> tuple[list[str], list[str], str]:
    """(labels, ids, selected_label) — при неизвестном id дописывает в начало."""
    options = list(STT_MODELS)
    if selected_id and selected_id not in _STT_BY_ID:
        options.insert(0, ModelOption(selected_id, "из .env"))
    labels = [m.label for m in options]
    ids = [m.id for m in options]
    label = next((m.label for m in options if m.id == selected_id), labels[0])
    return labels, ids, label


def chat_options_for_ui(selected_id: str) -> tuple[list[str], list[str], str]:
    options = list(CHAT_MODELS)
    if selected_id and selected_id not in _CHAT_BY_ID:
        options.insert(0, ModelOption(selected_id, "из .env"))
    labels = [m.label for m in options]
    ids = [m.id for m in options]
    label = next((m.label for m in options if m.id == selected_id), labels[0])
    return labels, ids, label


def id_from_stt_label(label: str, ids: list[str], labels: list[str]) -> str:
    try:
        idx = labels.index(label)
        return ids[idx]
    except ValueError:
        return label


def id_from_chat_label(label: str, ids: list[str], labels: list[str]) -> str:
    try:
        idx = labels.index(label)
        return ids[idx]
    except ValueError:
        return label
