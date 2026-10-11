"""ClipsDex command line interface."""

from __future__ import annotations

import typer

from clipsdex_engine import __version__, binaries

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
    version: bool = typer.Option(
        False, "--version", callback=_version_callback, is_eager=True, help="Show version."
    ),
) -> None:
    """ClipsDex: local, private analysis of your saved short videos."""


@app.command()
def doctor() -> None:
    """Check that the external tools ClipsDex needs are installed."""
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
