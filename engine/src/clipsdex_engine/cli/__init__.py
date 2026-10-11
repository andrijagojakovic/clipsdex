"""ClipsDex command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from clipsdex_engine import __version__, binaries, config, db
from clipsdex_engine.sources import manual

app = typer.Typer(
    name="clipsdex",
    help="ClipsDex: local, private analysis of your saved short videos.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"clipsdex {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """ClipsDex: local, private analysis of your saved short videos."""


@app.command("import")
def import_(
    paths: Annotated[list[Path], typer.Argument(help="Video files and/or folders to import.")],
    recursive: Annotated[bool, typer.Option(help="Also look inside subfolders.")] = True,
) -> None:
    """Import video files from your computer. Files already in the library are skipped."""
    engine = db.open_database()
    report = manual.import_paths(engine, paths, recursive=recursive)

    for path, reason in report.failed:
        typer.secho(f"[FAILED]  {path}: {reason}", fg=typer.colors.RED)
    typer.echo(
        f"Imported {len(report.added)} new video(s), "
        f"skipped {len(report.duplicates)} already in the library."
    )
    if not report.added and not report.duplicates and not report.failed:
        typer.echo(f"No video files found. Supported: {', '.join(sorted(manual.VIDEO_EXTENSIONS))}")
    if report.failed:
        raise typer.Exit(code=1)


@app.command()
def doctor() -> None:
    """Check that the external tools ClipsDex needs are installed."""
    typer.echo(f"Data folder: {config.data_dir()}\n")
    missing = 0
    for name in binaries.BINARIES:
        resolved = binaries.resolve(name)
        if resolved.path is None:
            missing += 1
            typer.secho(f"[MISSING] {name}: {resolved.problem}", fg=typer.colors.YELLOW)
            typer.echo(f"          {binaries.INSTALL_HINTS[name]}")
            continue
        typer.secho(f"[OK]      {name} ({resolved.source}): {resolved.path}", fg=typer.colors.GREEN)
        ver = binaries.version(resolved.path)
        typer.echo(f"          {ver or 'found, but it did not report a version'}")

    if missing:
        typer.secho(f"\n{missing} tool(s) missing.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    typer.secho("\nEverything needed is installed.", fg=typer.colors.GREEN)
