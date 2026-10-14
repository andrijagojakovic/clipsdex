"""Which movie a clip comes from (run comes in 4.4).

Only text, dialogue and scene descriptions are used; faces are never used to recognize
anyone.
"""

from __future__ import annotations

from clipsdex_engine.analyzers.base import Analyzer
from clipsdex_engine.config import ModelChoice


class MovieId(Analyzer):
    name = "movie_id"
    title = "Identify movies"
    requires = ("ocr", "transcript", "frame_descriptions")
    default_categories = ("movie",)
    available = False

    def resource_key(self, models: ModelChoice) -> str:
        return f"ollama:{models.text}"
