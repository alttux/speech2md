import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from speech2md.cli.commands import config_cmd
from speech2md.cli.commands.listen import listen_cmd
from speech2md.core.models import FormatMode
from speech2md.core.pipeline import run_file_job

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
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show debug logs"),
) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

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

        job = run_file_job(
            audio_path,
            language=language,
            use_llm=not no_llm,
            mode=mode,
            on_progress=on_progress,
        )

    for warning in job.warnings:
        console.print(f"[yellow]! {warning}[/yellow]")

    if job.status.value == "completed":
        assert job.result is not None
        console.print("\n[green]✓ Transcription saved[/green]")
        console.print(f"  File:     {job.output_path}")
        console.print(f"  Duration: {job.result.duration:.1f}s")
        console.print(f"  Language: {job.result.language or 'detected'}")
        console.print(f"  Words:    ~{len(job.result.full_text.split())}")
    else:
        console.print(f"\n[red]✗ Failed: {job.error}[/red]")
        if not verbose:
            console.print("[dim]Run with --verbose for a full traceback.[/dim]")
        raise typer.Exit(code=1)


app.add_typer(config_cmd.app, name="config", help="View or change configuration")
app.command(name="listen")(listen_cmd)
