"""Full categorization from the transcript, frames and on-screen text (run comes in 4.2)."""

from __future__ import annotations

from clipsdex_engine.analyzers.base import Analyzer
from clipsdex_engine.config import ModelChoice


class Categorize(Analyzer):
    name = "categorize"
    title = "Categorize"
    requires = ("transcript", "ocr", "frame_descriptions")
    default_categories = ()  # re-checks any video
    available = False

    def resource_key(self, models: ModelChoice) -> str:
        return f"ollama:{models.text}"
