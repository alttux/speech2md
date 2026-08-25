from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

from speech2md.core.models import TranscriptionResult

ProgressCallback = Callable[[float], None]


class Transcriber(ABC):
    @abstractmethod
    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "",
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        raise NotImplementedError
