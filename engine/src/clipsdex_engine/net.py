"""The one HTTP client for all external calls.

Timeouts, retries with backoff and a User-Agent naming the project. Never log secrets.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from clipsdex_engine import __version__

log = logging.getLogger(__name__)

USER_AGENT = f"ClipsDex/{__version__} (+https://github.com/andrijagojakovic/clipsdex)"
TIMEOUT = httpx.Timeout(30.0, connect=10.0)
RETRIES = 3
_CHUNK = 1024 * 1024

# Called with (bytes downloaded so far, total bytes or None).
Progress = Callable[[int, int | None], None]


class DownloadError(RuntimeError):
    """A download failed. The message is safe to show to users."""


def client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True)


@dataclass(frozen=True)
class Checksum:
    """Expected hash of a download. ``git-sha1`` is a git blob id (Hugging Face small files)."""

    kind: Literal["md5", "sha256", "git-sha1"]
    value: str
    size: int | None = None  # required for git-sha1 (part of the hashed header)

    def new_digest(self) -> Any:
        if self.kind == "git-sha1":
            if self.size is None:
                raise ValueError("A git-sha1 checksum needs the file size.")
            digest = hashlib.sha1(usedforsecurity=False)
            digest.update(f"blob {self.size}\0".encode())
            return digest
        return hashlib.new(self.kind, usedforsecurity=False)


def download(
    url: str,
    dst: Path,
    *,
    md5: str | None = None,
    checksum: Checksum | None = None,
    progress: Progress | None = None,
    retries: int = RETRIES,
) -> Path:
    """Download ``url`` to ``dst`` atomically, verifying ``checksum`` (or ``md5``) if given."""
    if md5 is not None:
        checksum = Checksum("md5", md5)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with client() as http, http.stream("GET", url) as response:
                response.raise_for_status()
                total = (
                    int(response.headers["Content-Length"])
                    if "Content-Length" in response.headers
                    else None
                )
                digest = checksum.new_digest() if checksum is not None else None
                done = 0
                with tmp.open("wb") as f:
                    for chunk in response.iter_bytes(_CHUNK):
                        f.write(chunk)
                        if digest is not None:
                            digest.update(chunk)
                        done += len(chunk)
                        if progress is not None:
                            progress(done, total)
            if checksum is not None and digest is not None and digest.hexdigest() != checksum.value:
                raise DownloadError("The downloaded file is damaged (checksum mismatch).")
            tmp.replace(dst)
            return dst
        except (httpx.HTTPError, DownloadError, OSError) as e:
            last_error = e
            tmp.unlink(missing_ok=True)
            log.warning("Download attempt %d/%d failed: %s", attempt, retries, e)
            if attempt < retries:
                time.sleep(2**attempt)
    raise DownloadError(
        f"Could not download {dst.name}. Check your internet connection and try again. "
        f"({last_error})"
    )
