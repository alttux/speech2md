from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from speech2md.core.config import get_output_dir
from speech2md.core.config import load as load_config
from speech2md.core.export import write_markdown
from speech2md.core.models import FormatMode, Job, JobMode, JobStatus, Settings
from speech2md.core.postprocess.formatter import process_text
from speech2md.core.postprocess.llm_client import create_client
from speech2md.core.stt.whisper_engine import WhisperTranscriber

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]

STT_PROGRESS_START = 0.05
STT_PROGRESS_END = 0.6
LLM_PROGRESS_END = 0.9


def _prompt_templates(settings: Settings, language: str, mode: FormatMode) -> tuple[str, str]:
    russian = language == "ru"
    if mode == FormatMode.conspect:
        template = (
            settings.prompt_template_conspect_ru
            if russian
            else settings.prompt_template_conspect_en
        )
    else:
        template = settings.prompt_template_ru if russian else settings.prompt_template_en
    merge = settings.prompt_template_merge_ru if russian else settings.prompt_template_merge_en
    return template, merge


def run_file_job(
    audio_path: str | Path,
    *,
    language: str = "",
    use_llm: bool = True,
    mode: FormatMode | str | None = None,
    on_progress: ProgressCallback | None = None,
) -> Job:
    settings = load_config()
    audio_path = Path(audio_path)
    format_mode = FormatMode(mode) if mode is not None else settings.format_mode

    if on_progress:
        on_progress("init", 0.0)

    job = Job(
        source=str(audio_path),
        mode=JobMode.file,
        language=language or settings.language,
        status=JobStatus.running,
    )

    try:
        if on_progress:
            on_progress("stt", STT_PROGRESS_START)

        transcriber = WhisperTranscriber(
            model_size=settings.whisper_model,
            device=settings.device,
            compute_type=settings.compute_type,
            beam_size=settings.beam_size,
        )

        def stt_progress(fraction: float) -> None:
            if on_progress:
                span = STT_PROGRESS_END - STT_PROGRESS_START
                on_progress("stt", STT_PROGRESS_START + span * fraction)

        result = transcriber.transcribe(
            audio_path,
            language=job.language,
            on_progress=stt_progress if on_progress else None,
        )
        job.language = job.language or result.language

        if on_progress:
            on_progress("stt", STT_PROGRESS_END)

        if use_llm:
            stage = "conspect" if format_mode == FormatMode.conspect else "llm"
            if on_progress:
                on_progress(stage, STT_PROGRESS_END)

            def llm_progress(fraction: float) -> None:
                if on_progress:
                    span = LLM_PROGRESS_END - STT_PROGRESS_END
                    on_progress(stage, STT_PROGRESS_END + span * fraction)

            try:
                llm = create_client(
                    backend=settings.llm_backend,
                    url=settings.llm_url,
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    timeout=settings.llm_timeout,
                    max_tokens=settings.llm_max_tokens,
                    retries=settings.llm_retries,
                    retry_backoff=settings.llm_retry_backoff,
                )
                template, merge_template = _prompt_templates(settings, job.language, format_mode)
                result.full_text = process_text(
                    result.full_text,
                    llm,
                    mode=format_mode,
                    language=job.language,
                    prompt_template=template,
                    merge_template=merge_template,
                    chunk_size=settings.chunk_size,
                    overlap=settings.overlap,
                    on_progress=llm_progress if on_progress else None,
                )
            except Exception as exc:
                message = f"LLM post-processing failed, keeping raw transcript: {exc}"
                logger.warning(message)
                job.warnings.append(message)

        if on_progress:
            on_progress("export", LLM_PROGRESS_END)

        suffix = ".conspect.md" if format_mode == FormatMode.conspect and use_llm else ".md"
        output_path = get_output_dir(settings) / f"{audio_path.stem}{suffix}"
        write_markdown(
            result,
            output_path,
            source=str(audio_path),
            language=result.language or job.language,
            model=settings.whisper_model,
        )

        job.result = result
        job.output_path = str(output_path)
        job.status = JobStatus.completed
        job.finished_at = datetime.now()

        if on_progress:
            on_progress("done", 1.0)

    except Exception as e:
        logger.exception("Job failed for %s", audio_path)
        job.status = JobStatus.failed
        job.error = f"{type(e).__name__}: {e}"
        job.finished_at = datetime.now()
        if on_progress:
            on_progress("error", 0.0)

    return job
