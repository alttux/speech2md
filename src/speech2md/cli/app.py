import logging
from pathlib import Path
from typing import Callable

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from speech2md.cli.commands import config_cmd
from speech2md.cli.commands.listen import listen_cmd
from speech2md.core.models import FormatMode, Job
from speech2md.core.pipeline import ProgressCallback, run_file_job, run_text_job

app = typer.Typer(name="speech2md")
console = Console()

STAGE_LABELS = {
    "init": "Initializing...",
    "stt": "Transcribing audio...",
    "llm": "Post-processing with LLM...",
    "conspect": "Building conspect...",
    "export": "Exporting...",
    "done": "Done!",
    "error": "Error!",
}


def _run_with_progress(run: Callable[[ProgressCallback], Job]) -> Job:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(STAGE_LABELS["init"], total=1.0)

        def on_progress(stage: str, percent: float) -> None:
            progress.update(
                task_id,
                description=STAGE_LABELS.get(stage, stage),
                completed=percent,
            )

        return run(on_progress)


def _report_job(job: Job, verbose: bool, *, verb: str = "saved") -> None:
    for warning in job.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")

    if job.status.value == "completed":
        assert job.result is not None
        console.print(f"\n[green]✓ {verb.capitalize()}[/green]")
        console.print(f"  File:     {job.output_path}")
        if job.result.duration:
            console.print(f"  Duration: {job.result.duration:.1f}s")
        console.print(f"  Language: {job.result.language or 'detected'}")
        console.print(f"  Words:    ~{len(job.result.full_text.split())}")
    else:
        console.print(f"\n[red]✗ Failed: {job.error}[/red]")
        if not verbose:
            console.print("[dim]Run with --verbose for a full traceback.[/dim]")
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    audio_path: Path = typer.Argument(..., help="Path to audio file (mp3, wav, etc.)", exists=True),
    language: str = typer.Option("", "--lang", "-l", help="Language code (ru, en, etc.)"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip LLM post-processing"),
    mode: FormatMode = typer.Option(
        None,
        "--mode",
        "-m",
        help="transcript = clean up the text, conspect = summarize into lecture notes",
    ),
    also_conspect: bool = typer.Option(
        False,
        "--also-conspect",
        help="After writing the plain transcript, also build a conspect from it (no second STT pass)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs"),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if also_conspect:
        if no_llm:
            console.print("[red]--also-conspect requires LLM post-processing; drop --no-llm.[/red]")
            raise typer.Exit(code=1)
        if mode == FormatMode.conspect:
            console.print(
                "[red]--also-conspect already builds a conspect; drop --mode conspect.[/red]"
            )
            raise typer.Exit(code=1)
        mode = mode or FormatMode.transcript

    job = _run_with_progress(
        lambda on_progress: run_file_job(
            audio_path,
            language=language,
            use_llm=not no_llm,
            mode=mode,
            on_progress=on_progress,
        )
    )
    _report_job(job, verbose, verb="transcription saved")

    if also_conspect and job.status.value == "completed":
        transcript_path = job.output_path
        assert transcript_path is not None
        console.print()
        conspect_job = _run_with_progress(
            lambda on_progress: run_text_job(
                transcript_path,
                language=job.language,
                mode=FormatMode.conspect,
                on_progress=on_progress,
            )
        )
        _report_job(conspect_job, verbose, verb="conspect saved")


@app.command()
def conspect(
    text_path: Path = typer.Argument(
        ..., help="Path to an already-transcribed text/markdown file", exists=True
    ),
    language: str = typer.Option("", "--lang", "-l", help="Language code (ru, en, etc.)"),
    mode: FormatMode = typer.Option(
        FormatMode.conspect,
        "--mode",
        "-m",
        help="conspect = summarize into lecture notes, transcript = just clean up the text",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs"),
) -> None:
    """Build a conspect (or cleaned transcript) from an existing text file, skipping STT."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    job = _run_with_progress(
        lambda on_progress: run_text_job(
            text_path,
            language=language,
            mode=mode,
            on_progress=on_progress,
        )
    )
    _report_job(job, verbose, verb="conspect saved")


app.add_typer(config_cmd.app, name="config", help="View or change configuration")
app.command(name="listen")(listen_cmd)
