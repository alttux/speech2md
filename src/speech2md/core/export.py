from __future__ import annotations

from datetime import datetime
from pathlib import Path

from speech2md.core.models import TranscriptionResult


def write_markdown(
    result: TranscriptionResult,
    output_path: str | Path,
    source: str = "",
    language: str = "",
    model: str = "",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    parts: list[str] = ["---"]
    parts.append(f"created: {datetime.now().isoformat()}")
    if source:
        parts.append(f"source: {source}")
    if language:
        parts.append(f"language: {language}")
    if model:
        parts.append(f"model: {model}")
    if result.duration:
        parts.append(f"duration_sec: {result.duration:.1f}")
    parts.append("---")
    parts.append("")
    parts.append(result.full_text)
    parts.append("")

    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def _srt_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(result: TranscriptionResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    entries = [
        f"{i}\n{_srt_timestamp(seg.start)} --> {_srt_timestamp(seg.end)}\n{seg.text}\n"
        for i, seg in enumerate(result.segments, start=1)
    ]

    path.write_text("\n".join(entries), encoding="utf-8")
    return path
