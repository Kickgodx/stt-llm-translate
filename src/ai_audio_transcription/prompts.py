"""Промпты: batch (полные) и live (короткие, меньше токенов)."""

# Речь в основном на русском, но с английскими IT/QA-терминами
CODE_SWITCHING_HINT = (
    "Основной язык речи — русский, но встречаются английские слова и термины "
    "(Python, SDET, automation, API, test, pytest и т.п.). "
    "Сохраняй такие термины латиницей; не транслитерируй (не «пайтон», «эсдет») "
    "и не переводи их, если пользователь не просил иное."
)

CODE_SWITCHING_HINT_SHORT = (
    "Russian speech with English tech terms — keep terms in Latin (Python, SDET, API)."
)

TRANSLATE_PROMPTS: dict[str, str] = {
    "en": (
        "Пользователь говорит на русском или другом языке, иногда вставляет английские термины. "
        "Переведи сказанное на английский. "
        f"{CODE_SWITCHING_HINT} "
        "Исправляй ошибочную транслитерацию в транскрипте (пайтон → Python). "
        "Выведи только перевод, без пояснений."
    ),
    "ru": (
        "The user speaks in English or another language, sometimes mixing Russian. "
        "Translate what was said into Russian. "
        "Keep English technical terms in Latin (Python, SDET, automation, API). "
        "Do not transliterate them into Cyrillic. "
        "Output only the translation, no commentary."
    ),
}

BATCH_LLM_SYSTEM = (
    "Ты обрабатываешь транскрипцию речи по инструкции пользователя. "
    f"{CODE_SWITCHING_HINT} "
    "Верни только итоговый текст без пояснений и метаданных."
)

# Минимальные промпты для лайв-режима (экономия токенов)
LIVE_LLM_SYSTEM = (
    "Translate the utterance. "
    f"{CODE_SWITCHING_HINT_SHORT} "
    "Output only the translation, nothing else."
)

LIVE_TRANSLATE_USER: dict[str, str] = {
    "en": "To English (keep tech terms in Latin):\n{text}",
    "ru": "To Russian (keep English tech terms in Latin):\n{text}",
}


def glossary_clause(glossary: str | None) -> str:
    """Дополнение к промпту со списком частых терминов из .env."""
    if not glossary or not glossary.strip():
        return ""
    return f" Частые термины в речи: {glossary.strip()}."


def batch_llm_system(*, glossary: str | None = None) -> str:
    return BATCH_LLM_SYSTEM + glossary_clause(glossary)


def live_llm_system(*, glossary: str | None = None) -> str:
    return LIVE_LLM_SYSTEM + glossary_clause(glossary)


def resolve_post_prompt(*, prompt: str | None, translate: str | None) -> str | None:
    if prompt and translate:
        raise ValueError("Укажите только --prompt или --translate, не оба сразу.")
    if prompt:
        return prompt
    if translate:
        key = translate.lower()
        if key not in TRANSLATE_PROMPTS:
            supported = ", ".join(sorted(TRANSLATE_PROMPTS))
            raise ValueError(
                f"Язык {translate!r} не поддержан для --translate. Доступно: {supported}"
            )
        return TRANSLATE_PROMPTS[key]
    return None


def resolve_live_post_prompt(*, prompt: str | None, translate: str | None) -> str | None:
    """Короткий user-message для LLM в лайв-режиме."""
    if prompt and translate:
        raise ValueError("Укажите только промпт или перевод, не оба сразу.")
    if prompt:
        return prompt.strip()
    if translate:
        key = translate.lower()
        if key not in LIVE_TRANSLATE_USER:
            supported = ", ".join(sorted(LIVE_TRANSLATE_USER))
            raise ValueError(f"Перевод {translate!r} недоступен. Доступно: {supported}")
        return LIVE_TRANSLATE_USER[key]
    return None


def build_live_user_content(post_prompt_template: str, transcript: str) -> str:
    if "{text}" in post_prompt_template:
        return post_prompt_template.format(text=transcript.strip())
    return f"{post_prompt_template}\n\n{transcript.strip()}"


def default_stt_prompt(*, glossary: str | None = None) -> str | None:
    """
    Подсказка для Whisper (если API провайдера её принимает).
    Короткий «стилевой» фрагмент с типичной лексикой улучшает распознавание терминов.
    """
    base = (
        "Разговор на русском языке. Термины: Python, SDET, automation, API, test, pytest."
    )
    extra = glossary_clause(glossary)
    if not extra:
        return base
    return base + extra
