from pathlib import Path

import pytest
from pydantic import ValidationError

from speech2md.core.config import get_output_dir
from speech2md.core.models import Settings


def _apply_cli_set(settings: Settings, key: str, value: str) -> Settings:
    """Mirrors the coercion done by `speech2md config set`."""
    data = settings.model_dump(mode="json")
    data[key] = value
    return Settings.model_validate(data)


def test_set_coerces_int_field() -> None:
    settings = Settings()
    updated = _apply_cli_set(settings, "beam_size", "10")
    assert updated.beam_size == 10
    assert isinstance(updated.beam_size, int)


def test_set_coerces_float_field() -> None:
    settings = Settings()
    updated = _apply_cli_set(settings, "vad_threshold", "0.7")
    assert updated.vad_threshold == 0.7
    assert isinstance(updated.vad_threshold, float)


def test_set_coerces_enum_field() -> None:
    settings = Settings()
    updated = _apply_cli_set(settings, "device", "cuda")
    assert updated.device.value == "cuda"


def test_set_coerces_format_mode() -> None:
    settings = Settings()
    updated = _apply_cli_set(settings, "format_mode", "conspect")
    assert updated.format_mode.value == "conspect"


def test_set_rejects_invalid_value() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        _apply_cli_set(settings, "beam_size", "not-a-number")


def test_get_output_dir_expands_tilde() -> None:
    output_dir = get_output_dir(Settings(output_dir="~/notes"))

    assert output_dir.is_absolute()
    assert "~" not in str(output_dir)
    assert output_dir == Path.home() / "notes"


def test_get_output_dir_keeps_relative_paths_relative() -> None:
    assert get_output_dir(Settings(output_dir="outputs")) == Path("outputs")
