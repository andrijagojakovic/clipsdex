"""Notes from advice videos, merged into one brief (run comes in 4.3)."""

from __future__ import annotations

from clipsdex_engine.analyzers.base import Analyzer, Applicability, not_profiled, pct
from clipsdex_engine.config import ModelChoice
from clipsdex_engine.db.models import Profile, Video

MIN_SPEECH = 0.2  # share of the audio that must be speech, if there is no on-screen text


class Brief(Analyzer):
    name = "brief"
    title = "Make brief"
    requires = ("transcript", "ocr")
    default_categories = ("business", "education")
    available = False

    def resource_key(self, models: ModelChoice) -> str:
        return f"ollama:{models.text}"

    def is_applicable(self, video: Video, profile: Profile | None) -> Applicability:
        if profile is None:
            return not_profiled()
        speech = profile.speech_ratio or 0.0
        if speech >= MIN_SPEECH:
            return Applicability(True, f"speech in {pct(speech)} of the audio")
        if profile.has_text:
            return Applicability(True, "has on-screen text")
        return Applicability(False, "no speech and no on-screen text to take notes from")
