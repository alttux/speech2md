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
    llm_model: str = ""
    llm_api_key: str = ""
    llm_timeout: int = 120

    prompt_template_ru: str = (
        "Исправь ошибки распознавания речи, не меняя смысл и стиль. "
        "Оформи текст как Markdown. Сохрани структуру речи.\n\n{text}"
    )
    prompt_template_en: str = (
        "Fix speech recognition errors without changing meaning or style. "
        "Format the text as Markdown. Preserve speech structure.\n\n{text}"
    )

    output_dir: str = ""
    vad_threshold: float = 0.5
    vad_min_speech_duration: float = 0.5
    vad_min_silence_duration: float = 1.0
