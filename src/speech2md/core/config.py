from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w
from platformdirs import user_config_dir

from speech2md.core.models import Settings

CONFIG_DIR = Path(user_config_dir("speech2md", ensure_exists=True))
CONFIG_FILE = CONFIG_DIR / "config.toml"

DEFAULT_OUTPUT_DIRNAME = "outputs"


def project_root() -> Path | None:
    """Repo root of an editable/source checkout, or None when installed as a wheel."""
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if not pyproject.is_file():
            continue
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            return None
        if data.get("project", {}).get("name") == "speech2md":
            return parent
        return None
    return None


def load() -> Settings:
    if not CONFIG_FILE.exists():
        settings = Settings()
    else:
        raw = CONFIG_FILE.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
        settings = Settings(**data)

    if not settings.llm_api_key:
        settings.llm_api_key = os.environ.get("OPENAI_API_KEY", "")
    if settings.llm_backend.value == "openai":
        settings.llm_url = os.environ.get("OPENAI_BASE_URL", settings.llm_url)
    elif settings.llm_backend.value == "ollama":
        settings.llm_url = os.environ.get("OLLAMA_URL", settings.llm_url)

    return settings


def save(settings: Settings) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = settings.model_dump(mode="json")
    CONFIG_FILE.write_text(tomli_w.dumps(raw), encoding="utf-8")


def get_output_dir(settings: Settings) -> Path:
    if settings.output_dir:
        # expanduser matters: without it a config value like "~/notes" is taken
        # literally and silently creates a directory named "~" under the cwd.
        return Path(settings.output_dir).expanduser()
    # Default to <project>/outputs so results sit next to the checkout; a wheel
    # install has no checkout, so fall back to the cwd.
    root = project_root() or Path.cwd()
    return root / DEFAULT_OUTPUT_DIRNAME
