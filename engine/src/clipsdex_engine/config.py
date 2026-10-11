"""Data directories, model presets and user settings.

User data lives in the platformdirs user data dir (``%LOCALAPPDATA%\\ClipsDex`` on Windows).
Set ``CLIPSDEX_DATA_DIR`` to use another folder (development, tests, portable setups).

Settings here are non-secret only; API keys go to ``keyring`` (see CLAUDE.md §8).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import platformdirs
from pydantic import BaseModel, ConfigDict, Field

APP_NAME = "ClipsDex"
DATA_DIR_ENV = "CLIPSDEX_DATA_DIR"


@dataclass(frozen=True)
class DataPaths:
    """All locations inside the data dir. Use ``ensure()`` before writing."""

    root: Path

    @property
    def db(self) -> Path:
        return self.root / "clipsdex.db"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def videos(self) -> Path:
        """Video files the app owns (e.g. downloaded); deleted after analysis by default."""
        return self.root / "videos"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> DataPaths:
        for folder in (self.root, self.artifacts, self.videos, self.logs):
            folder.mkdir(parents=True, exist_ok=True)
        return self


def data_dir() -> Path:
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path(platformdirs.user_data_dir(APP_NAME, appauthor=False, roaming=False))


def data_paths() -> DataPaths:
    return DataPaths(data_dir())


class Preset(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    DETAILED = "detailed"


class ModelChoice(BaseModel):
    """Models and frame budget for one preset (CLAUDE.md §4)."""

    model_config = ConfigDict(frozen=True)

    whisper: str
    vision: str
    text: str
    embedding: str
    max_frames: int = Field(ge=1)


# Ollama tags must be verified against the Ollama library before use (see CLAUDE.md §15).
PRESETS: dict[Preset, ModelChoice] = {
    Preset.FAST: ModelChoice(
        whisper="base",
        vision="moondream",
        text="llama3.2:3b",
        embedding="nomic-embed-text",
        max_frames=2,
    ),
    Preset.BALANCED: ModelChoice(
        whisper="small",
        vision="gemma3:4b",
        text="gemma3:4b",
        embedding="nomic-embed-text",
        max_frames=4,
    ),
    Preset.DETAILED: ModelChoice(
        whisper="medium",
        vision="qwen2.5vl:7b",
        text="qwen2.5:7b",
        embedding="nomic-embed-text",
        max_frames=8,
    ),
}


class Settings(BaseModel):
    """User settings with safe defaults. ``None`` overrides mean "use the preset"."""

    model_config = ConfigDict(validate_assignment=True)

    preset: Preset = Preset.BALANCED
    max_frames: int | None = Field(default=None, ge=1, le=32)
    whisper_model: str | None = None
    vision_model: str | None = None
    text_model: str | None = None
    embedding_model: str | None = None
    delete_after_analysis: bool = True  # D10
    downloader_enabled: bool = False  # D9: only after the consent screen

    def models(self) -> ModelChoice:
        """The preset's models with any user overrides applied."""
        base = PRESETS[self.preset]
        return ModelChoice(
            whisper=self.whisper_model or base.whisper,
            vision=self.vision_model or base.vision,
            text=self.text_model or base.text,
            embedding=self.embedding_model or base.embedding,
            max_frames=self.max_frames or base.max_frames,
        )
