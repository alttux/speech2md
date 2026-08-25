from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from speech2md.core.models import Device, Segment, TranscriptionResult
from speech2md.core.stt.base import ProgressCallback, Transcriber


class WhisperTranscriber(Transcriber):
    def __init__(
        self,
        model_size: str = "small",
        device: Device | str = Device.cpu,
        compute_type: str = "int8",
        beam_size: int = 5,
        cache_dir: Path | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = Device(device) if isinstance(device, str) else device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.cache_dir = cache_dir
        self._model: WhisperModel | None = None

    @property
    def model(self) -> WhisperModel:
        if self._model is None:
            self._model = WhisperModel(
                self.model_size,
                device=self.device.value,
                compute_type=self.compute_type,
                download_root=str(self.cache_dir) if self.cache_dir else None,
            )
        return self._model

    def transcribe(
        self,
        audio_path: str | Path,
        language: str = "",
        on_progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        segments_gen, info = self.model.transcribe(str(audio_path), language=language or None, beam_size=self.beam_size)

        segments: list[Segment] = []
        full_text_parts: list[str] = []
        duration = float(getattr(info, "duration", 0.0) or 0.0)

        for fw_seg in segments_gen:
            segments.append(
                Segment(
                    start=fw_seg.start,
                    end=fw_seg.end,
                    text=fw_seg.text.strip(),
                    language="",
                    confidence=fw_seg.avg_logprob,
                )
            )
            full_text_parts.append(fw_seg.text.strip())
            if on_progress and duration > 0:
                on_progress(min(1.0, fw_seg.end / duration))

        if on_progress:
            on_progress(1.0)

        return TranscriptionResult(
            segments=segments,
            full_text=" ".join(full_text_parts),
            language=getattr(info, "language", ""),
            duration=duration,
        )
