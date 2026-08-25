from speech2md.core.models import FormatMode, Settings
from speech2md.core.postprocess.prompts import (
    PROMPT_CONSPECT_EN,
    PROMPT_CONSPECT_RU,
    PROMPT_EN,
    PROMPT_RU,
    get_prompt,
)


def test_get_prompt_defaults_to_transcript_mode() -> None:
    assert get_prompt("ru") == PROMPT_RU
    assert get_prompt("en") == PROMPT_EN


def test_get_prompt_conspect_mode() -> None:
    assert get_prompt("ru", mode="conspect") == PROMPT_CONSPECT_RU
    assert get_prompt("en", mode="conspect") == PROMPT_CONSPECT_EN


def test_get_prompt_unknown_language_falls_back_to_english() -> None:
    assert get_prompt("fr") == PROMPT_EN
    assert get_prompt("fr", mode="conspect") == PROMPT_CONSPECT_EN


def test_get_prompt_override_wins_regardless_of_mode() -> None:
    assert get_prompt("ru", "custom {text}", mode="conspect") == "custom {text}"


def test_settings_default_format_mode_is_transcript() -> None:
    settings = Settings()
    assert settings.format_mode == FormatMode.transcript
    assert "{text}" in settings.prompt_template_conspect_ru
    assert "{text}" in settings.prompt_template_conspect_en
