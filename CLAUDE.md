# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

speech2md transcribes audio to text (faster-whisper) and post-processes the transcript through an LLM (Ollama or OpenAI-compatible API) to clean up recognition errors and format it as Markdown. Currently ships as a CLI (Typer); a GTK GUI (`src/speech2md/gui/`) and a realtime microphone mode (`speech2md listen`) are stubbed but not implemented.

## Цель проекта:
Переводить аудио файлы в md записи лекций, а после составлять конспекты

## Commands

Install (editable, with CLI + dev extras):
```
pip install -e ".[cli,dev]"
```
Other optional extras: `gui` (PyGObject), `realtime` (sounddevice/silero-vad/onnxruntime), `cuda` (GPU whisper/onnxruntime).

Run the CLI:
```
speech2md transcribe <audio_path> [--lang ru|en] [--no-llm] [--mode transcript|conspect] [--verbose]
speech2md config show
speech2md config set <key> <value>
speech2md config path
```

Tests (pytest, configured via `pyproject.toml` with `pythonpath = ["src"]`, so no install needed to run tests):
```
pytest                          # full suite
pytest tests/test_models.py::test_segment   # single test
```

Lint / type-check:
```
ruff check .
mypy src
```

## Architecture

Package root: `src/speech2md/`, split into `core`, `cli`, `gui`.

**Pipeline flow** (`core/pipeline.py::run_file_job`) is the central orchestrator, called by the CLI's `transcribe` command:
1. Load `Settings` (`core/config.py::load`).
2. Transcribe audio via `core/stt/whisper_engine.py::WhisperTranscriber` (wraps `faster_whisper.WhisperModel`, lazily instantiated on first use) into a `TranscriptionResult` (`core/models.py`).
3. Unless `--no-llm`, post-process the full text via `core/postprocess/formatter.py::process_text`, which dispatches on `FormatMode`:
   - **transcript** (`format_long_text`) — cleans up recognition errors chunk by chunk, then stitches chunks back together by detecting/trimming duplicated overlap text between consecutive outputs (`_find_overlap`, which ignores matches shorter than `MIN_OVERLAP_MATCH` to avoid false positives).
   - **conspect** (`summarize_long_text`) — map-reduce summarization: each chunk is summarized on its own, then the partial summaries are merged into one document by a final LLM pass (`_merge_summaries`, using the `merge` prompt). If the summaries are too large to merge in one pass they are batched by `_group_by_budget` and merged over repeated rounds; if even that can't shrink them they are joined verbatim with a logged warning.

   Chunking (`build_chunks`) is aligned to sentence boundaries and budgeted in **words** (`chunk_size`/`overlap` in `Settings`); `overlap` is clamped to `chunk_size / 2`, and sentences longer than `chunk_size` (common with sparsely punctuated Whisper output) are hard-wrapped. LLM failures are caught, logged, and appended to `job.warnings` — the pipeline still completes with the raw transcript rather than failing the job.
4. Export to Markdown via `core/export.py::write_markdown`, writing a YAML frontmatter block (created/source/language/model/duration) followed by the text, to `<output_dir>/<audio_stem>.md` (or `<audio_stem>.conspect.md` in conspect mode). `write_srt` exports timestamped segments as SubRip; it is not yet wired into the pipeline.
5. Progress is reported through an `on_progress(stage, percent)` callback with stages `init/stt/llm/conspect/export/done/error`, consumed by the CLI to drive a Rich progress bar. STT reports fine-grained progress from segment end times against the audio duration, and the LLM stages report per-chunk fractions, so long files show real movement instead of a stuck spinner.

**STT abstraction**: `core/stt/base.py::Transcriber` is an ABC; `WhisperTranscriber` is the only implementation. Add new engines here if needed.

**LLM abstraction**: `core/postprocess/llm_client.py::LLMClient` is an ABC with `OllamaClient` (`/api/generate`, non-chat) and `OpenAIClient` (`/chat/completions`) implementations, selected via `create_client(backend=...)`. Both post through `_post_with_retry`, which retries timeouts, transport errors and retryable statuses (`RETRYABLE_STATUS`) with exponential backoff (`llm_retries`/`llm_retry_backoff` in `Settings`); 4xx responses are returned as-is so callers can special-case them (e.g. Ollama's 404 "model not pulled" message).

Prompts come from `core/postprocess/prompts.py`, keyed by mode then language in the `PROMPTS` dict (`transcript`/`conspect`/`merge` × `ru`/`en`), but are normally overridden by the matching `prompt_template_*` field in `Settings`, chosen based on the job's language. `get_prompt(language, template_override, *, mode=...)` falls back to English for unrecognized languages and to `transcript` for unrecognized modes; an explicit override always wins.

**Config** (`core/config.py`): `Settings` (pydantic model in `core/models.py`) is loaded from/saved to a single TOML file at the platformdirs user-config location (`~/.config/speech2md/config.toml` on Linux). `config.example.toml` documents every field (in Russian) and is the reference for adding new settings. On load, `llm_api_key` falls back to the `OPENAI_API_KEY` env var if unset, and `llm_url` is overridden by `OPENAI_BASE_URL` or `OLLAMA_URL` depending on `llm_backend` — env vars win over the TOML file for those specific fields. When adding a new setting: add the field to `Settings`, document it in `config.example.toml`, and (if it should be env-overridable) wire it into `load()`.

**Models** (`core/models.py`): all data types are pydantic `BaseModel`s — `Segment`, `TranscriptionResult`, `Job` (with `JobStatus`/`JobMode` enums), `Settings` (with `LLMBackend`/`Device`/`FormatMode` enums). `Job` carries `result: TranscriptionResult | None` and `error: str | None` for pass/fail reporting, plus `warnings: list[str]` for non-fatal degradations (e.g. LLM unreachable) and `output_path` for the written file.

**CLI** (`cli/app.py`): Typer app; entry point is `speech2md = "speech2md.cli.app:app"` in `pyproject.toml`. Subcommands live in `cli/commands/` (`config_cmd.py`, `listen.py`) and are mounted onto the root app. `listen` is currently a no-op stub pending the realtime pipeline.

`config set` coerces values by round-tripping through `Settings.model_validate` (CLI arguments always arrive as strings) and reports a `ValidationError` instead of storing a wrong-typed value — do not use `setattr` on `Settings`, which would silently skip validation.

**GUI** (`gui/`): package and `gui/pages/` exist but are currently empty — no implementation yet.
