PROMPT_RU = (
    "Ты — помощник по очистке распознанной речи.\n"
    "Исправь только очевидные ошибки распознавания (неправильные окончания, "
    "потерянные слова, неверные термины). Не меняй смысл, стиль и структуру речи.\n"
    "Оформи результат как Markdown. Сохрани абзацы и интонационные паузы как пустые строки.\n"
    "Не добавляй ничего от себя.\n\n"
    "{text}"
)

PROMPT_EN = (
    "You are a speech-recognition cleanup assistant.\n"
    "Fix only obvious recognition errors (wrong endings, missing words, incorrect terms). "
    "Do not change the meaning, style, or structure of the speech.\n"
    "Format the output as Markdown. Preserve paragraphs and pauses as blank lines.\n"
    "Do not add anything extra.\n\n"
    "{text}"
)

PROMPT_CONSPECT_RU = (
    "Ты — помощник, который делает конспекты лекций.\n"
    "Составь конспект по расшифровке ниже: выдели темы заголовками Markdown, "
    "ключевые идеи — маркированными списками.\n"
    "Сохрани термины, определения, формулы и примеры дословно.\n"
    "Опусти оговорки, повторы и слова-паразиты. Не добавляй того, чего нет в тексте.\n\n"
    "{text}"
)

PROMPT_CONSPECT_EN = (
    "You are a lecture note-taking assistant.\n"
    "Write notes from the transcript below: use Markdown headings for topics and "
    "bullet lists for key ideas.\n"
    "Keep terms, definitions, formulas and examples verbatim.\n"
    "Drop filler, repetitions and false starts. Do not add anything not in the text.\n\n"
    "{text}"
)

PROMPT_MERGE_RU = (
    "Ниже — конспекты последовательных частей одной лекции, разделённые маркерами.\n"
    "Объедини их в один связный конспект Markdown: слей дублирующиеся разделы, "
    "убери повторы, выстрой сквозную структуру заголовков без нумерации частей.\n"
    "Сохрани все термины, определения и примеры. Не добавляй ничего от себя.\n\n"
    "{text}"
)

PROMPT_MERGE_EN = (
    "Below are notes from consecutive parts of one lecture, separated by markers.\n"
    "Merge them into a single coherent Markdown document: fold duplicate sections "
    "together, remove repetitions, and build one consistent heading structure without "
    "part numbering.\n"
    "Keep all terms, definitions and examples. Do not add anything extra.\n\n"
    "{text}"
)

PROMPTS: dict[str, dict[str, str]] = {
    "transcript": {"ru": PROMPT_RU, "en": PROMPT_EN},
    "conspect": {"ru": PROMPT_CONSPECT_RU, "en": PROMPT_CONSPECT_EN},
    "merge": {"ru": PROMPT_MERGE_RU, "en": PROMPT_MERGE_EN},
}


def get_prompt(
    language: str = "",
    template_override: str = "",
    *,
    mode: str = "transcript",
) -> str:
    if template_override:
        return template_override
    by_language = PROMPTS.get(mode, PROMPTS["transcript"])
    return by_language.get(language, by_language["en"])
