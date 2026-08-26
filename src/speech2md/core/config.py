from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w
from platformdirs import user_config_dir

from speech2md.core.models import Settings

CONFIG_DIR = Path(user_config_dir("speech2md", ensure_exists=True))
CONFIG_FILE = CONFIG_DIR / "config.toml"


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
    return Path(user_config_dir("speech2md", ensure_exists=True)) / "output"
