"""Locate the external binaries the engine needs (ffmpeg, ffprobe, fpcalc).

Every part of the engine resolves binaries through this module; never hardcode paths.

Lookup order for each binary:
1. Its own env var override, e.g. ``CLIPSDEX_FFMPEG=C:\\tools\\ffmpeg.exe``.
2. The bundled binaries folder: ``CLIPSDEX_BIN_DIR`` (set by the desktop app).
3. The system ``PATH``.

If an override is set but points to a missing file, the binary is reported as missing
instead of silently falling back, so a typo in the override does not go unnoticed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

BINARIES: tuple[str, ...] = ("ffmpeg", "ffprobe", "fpcalc")

BIN_DIR_ENV = "CLIPSDEX_BIN_DIR"

INSTALL_HINTS: dict[str, str] = {
    "ffmpeg": "Install it with: winget install Gyan.FFmpeg",
    "ffprobe": "It comes with ffmpeg. Install it with: winget install Gyan.FFmpeg",
    "fpcalc": "Install Chromaprint with: winget install AcoustID.Chromaprint",
}


class Source(StrEnum):
    ENV = "env"
    BUNDLED = "bundled"
    PATH = "path"


@dataclass(frozen=True)
class ResolvedBinary:
    name: str
    path: Path | None
    source: Source | None
    problem: str | None = None

    @property
    def found(self) -> bool:
        return self.path is not None


class BinaryNotFoundError(RuntimeError):
    """A required binary could not be found. The message is safe to show to users."""


def override_env_var(name: str) -> str:
    return f"CLIPSDEX_{name.upper()}"


def _executable_name(name: str) -> str:
    return f"{name}.exe" if sys.platform == "win32" else name


def resolve(name: str) -> ResolvedBinary:
    """Find ``name`` using the lookup order described in the module docstring."""
    override = os.environ.get(override_env_var(name))
    if override:
        path = Path(override)
        if path.is_file():
            return ResolvedBinary(name, path, Source.ENV)
        return ResolvedBinary(
            name,
            None,
            None,
            problem=f"{override_env_var(name)} is set to {path}, but that file does not exist.",
        )

    bin_dir = os.environ.get(BIN_DIR_ENV)
    if bin_dir:
        path = Path(bin_dir) / _executable_name(name)
        if path.is_file():
            return ResolvedBinary(name, path, Source.BUNDLED)

    on_path = shutil.which(name)
    if on_path:
        return ResolvedBinary(name, Path(on_path), Source.PATH)

    return ResolvedBinary(name, None, None, problem=f"{name} was not found.")


def require(name: str) -> Path:
    """Return the path to ``name`` or raise a user-readable ``BinaryNotFoundError``."""
    resolved = resolve(name)
    if resolved.path is None:
        raise BinaryNotFoundError(f"{resolved.problem} {INSTALL_HINTS.get(name, '')}".strip())
    return resolved.path


def version(path: Path, timeout: float = 10.0) -> str | None:
    """Return the first line of ``<binary> -version``, or None if it cannot be run."""
    try:
        result = subprocess.run(
            [str(path), "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else None
