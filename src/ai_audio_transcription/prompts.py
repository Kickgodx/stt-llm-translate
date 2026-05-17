"""Промпты: batch (полные) и live (короткие, меньше токенов)."""

TRANSLATE_PROMPTS: dict[str, str] = {
    "en": (
        "Пользователь говорит на русском или другом языке. "
        "Переведи сказанное на английский. Выведи только перевод, без пояснений."
    ),
    "ru": (
        "The user speaks in English or another language. "
        "Translate what was said into Russian. Output only the translation, no commentary."
    ),
}

BATCH_LLM_SYSTEM = (
    "Ты обрабатываешь транскрипцию речи по инструкции пользователя. "
    "Верни только итоговый текст без пояснений и метаданных."
)

# Минимальные промпты для лайв-режима (экономия токенов)
LIVE_LLM_SYSTEM = "Translate the utterance. Output only the translation, nothing else."

LIVE_TRANSLATE_USER: dict[str, str] = {
    "en": "To English:\n{text}",
    "ru": "To Russian:\n{text}",
}


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
