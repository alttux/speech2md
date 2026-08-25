from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Segment(BaseModel):
    start: float
    end: float
    text: str
    language: str = "ru"
    confidence: float = 0.0


class TranscriptionResult(BaseModel):
    segments: list[Segment]
    full_text: str
    language: str = ""
    duration: float = 0.0


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobMode(str, Enum):
    file = "file"
    realtime = "realtime"


class FormatMode(str, Enum):
    transcript = "transcript"
    conspect = "conspect"


class Job(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str
    mode: JobMode
    status: JobStatus = JobStatus.pending
    language: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    finished_at: datetime | None = None
    result: TranscriptionResult | None = None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    output_path: str | None = None


class LLMBackend(str, Enum):
    ollama = "ollama"
    openai = "openai"


class Device(str, Enum):
    cpu = "cpu"
    cuda = "cuda"


class Settings(BaseModel):
    whisper_model: str = "small"
    device: Device = Device.cpu
    compute_type: str = "int8"
    language: str = ""
    beam_size: int = 5

    llm_backend: LLMBackend = LLMBackend.ollama
    llm_url: str = "http://localhost:11434"
    llm_model: str = "gemma4"
    llm_api_key: str = ""
    llm_timeout: int = 120
    llm_retries: int = 2
    llm_retry_backoff: float = 2.0

    format_mode: FormatMode = FormatMode.transcript

    prompt_template_ru: str = (
        "Исправь ошибки распознавания речи, не меняя смысл и стиль. "
        "Оформи текст как Markdown. Сохрани структуру речи.\n\n{text}"
    )
    prompt_template_en: str = (
        "Fix speech recognition errors without changing meaning or style. "
        "Format the text as Markdown. Preserve speech structure.\n\n{text}"
    )
    prompt_template_conspect_ru: str = (
        "Составь конспект лекции по расшифровке ниже. Выдели темы заголовками Markdown, "
        "ключевые идеи — списками, сохрани термины, определения и примеры. "
        "Опусти оговорки и повторы.\n\n{text}"
    )
    prompt_template_conspect_en: str = (
        "Write lecture notes from the transcript below. Use Markdown headings for topics, "
        "bullet lists for key ideas, and keep terms, definitions and examples. "
        "Drop filler and repetitions.\n\n{text}"
    )
    prompt_template_merge_ru: str = (
        "Ниже — конспекты последовательных частей одной лекции. Объедини их в один связный "
        "конспект Markdown: убери повторы, слей дублирующиеся разделы, выстрой сквозную "
        "структуру заголовков. Не добавляй информацию, которой нет в тексте.\n\n{text}"
    )
    prompt_template_merge_en: str = (
        "Below are notes from consecutive parts of one lecture. Merge them into a single "
        "coherent Markdown document: remove repetitions, fold duplicate sections together, "
        "and build one consistent heading structure. Do not add information that is not "
        "in the text.\n\n{text}"
    )

    llm_max_tokens: int = 0

    chunk_size: int = 2000
    overlap: int = 200

    output_dir: str = ""
    vad_threshold: float = 0.5
    vad_min_speech_duration: float = 0.5
    vad_min_silence_duration: float = 1.0
