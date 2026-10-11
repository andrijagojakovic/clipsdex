"""Artifacts: cached intermediate results shared between analyzers (CLAUDE.md §5.2).

An artifact is identified by (video, type, params hash). Files live under
``<data dir>/artifacts/<video id>/`` and are stored in the DB relative to the data dir,
so the whole data folder can be moved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlmodel import Session, select

from clipsdex_engine.config import DataPaths
from clipsdex_engine.db.models import Artifact


def params_hash(params: dict[str, Any]) -> str:
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def artifact_dir(paths: DataPaths, video_id: int) -> Path:
    return paths.artifacts / str(video_id)


def to_stored_path(paths: DataPaths, path: Path) -> str:
    return path.relative_to(paths.root).as_posix()


def from_stored_path(paths: DataPaths, stored: str) -> Path:
    return paths.root / stored


def find(session: Session, video_id: int, type_: str, params: dict[str, Any]) -> Artifact | None:
    return session.exec(
        select(Artifact).where(
            Artifact.video_id == video_id,
            Artifact.type == type_,
            Artifact.params_hash == params_hash(params),
        )
    ).first()
