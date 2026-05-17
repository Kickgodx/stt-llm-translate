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
