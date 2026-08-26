import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from speech2md.core.config import CONFIG_FILE
from speech2md.core.config import load as load_config
from speech2md.core.config import save as save_config
from speech2md.core.models import Settings

app = typer.Typer()
console = Console()


@app.command()
def show() -> None:
    settings = load_config()
    table = Table("Key", "Value")
    for key, value in settings.model_dump(mode="json").items():
        table.add_row(key, str(value))
    console.print(table)


@app.command()
def set(
    key: str = typer.Argument(..., help="Setting key"),
    value: str = typer.Argument(..., help="Setting value"),
) -> None:
    settings = load_config()
    if key not in Settings.model_fields:
        console.print(f"[red]Unknown setting: {key}[/red]")
        raise typer.Exit(code=1)

    data = settings.model_dump(mode="json")
    data[key] = value
    try:
        updated = Settings.model_validate(data)
    except ValidationError as exc:
        console.print(f"[red]Invalid value for {key}: {value}[/red]")
        for err in exc.errors():
            console.print(f"  {err['msg']}")
        raise typer.Exit(code=1) from exc

    save_config(updated)
    stored = updated.model_dump(mode="json")[key]
    console.print(f"[green]{key}[/green] set to [yellow]{stored}[/yellow]")
    console.print(f"Saved to {CONFIG_FILE}")


@app.command()
def path() -> None:
    console.print(str(CONFIG_FILE))
