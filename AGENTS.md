# speech2md

**Speech-to-text with LLM post-processing, exporting to Markdown.**

## Quick start

```bash
pip install -e ".[dev,cli]"
speech2md --help                  # CLI entrypoint: speech2md.cli.app:app
speech2md transcribe audio.mp3
speech2md config show             # TOML config in platformdirs user_config_dir
speech2md config set whisper_model large
speech2md listen                  # stub — not yet implemented
```

Install extras: `.[gui]` (PyGObject), `.[realtime]` (sounddevice + silero-vad + onnxruntime), `.[cuda]`.

## Verification commands

```bash
ruff check src tests              # no formatter, lint only
mypy src tests                    # strict mode
pytest -v tests/                  # single test file (test_models.py)
```

No lockfile — run `pip install -e ".[dev,cli]"` after clone.

## Architecture

- **`src/speech2md/core/`** — domain logic: models (Pydantic), config (TOML), pipeline, export, STT (faster-whisper), LLM post-processing (Ollama / OpenAI)
- **`src/speech2md/cli/`** — Typer CLI (`app.py`), config commands, listen stub
- **`src/speech2md/gui/`** — PyGObject GUI skeleton (pages placeholder, no implementation)
- **`tests/test_models.py`** — only existing tests; no integration or snapshot tests

## Key details

- Python >= 3.11, strict mypy, Ruff (E/F/I/N/W, line-length 100, no E501)
- Default LLM: Ollama `gemma4` at `http://localhost:11434`; also supports OpenAI-compatible
- Default Whisper model: `small`, compute `int8`, device `cpu`
- Output: Markdown with YAML frontmatter (created, source, language, model, duration_sec)
- `realtime` and `gui` features are stubs — live mic transcription and GUI not implemented
- `data/` directory is empty — no dataset or migrations
- Single commit, early-stage project
