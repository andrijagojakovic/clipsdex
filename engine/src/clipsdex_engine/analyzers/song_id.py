"""Which song an edit uses (run comes in 4.5-4.7)."""

from __future__ import annotations

from clipsdex_engine.analyzers.base import Analyzer, Applicability, not_profiled, pct
from clipsdex_engine.db.models import Profile, Video

MIN_MUSIC = 0.15  # share of the audio that must be music


class SongId(Analyzer):
    name = "song_id"
    title = "Find songs"
    # The audio track is extracted during import; later steps may add OCR and vocals.
    requires = ()
    default_categories = ("edit", "music")
    available = False

    def is_applicable(self, video: Video, profile: Profile | None) -> Applicability:
        if profile is None:
            return not_profiled()
        if profile.has_audio is False:
            return Applicability(False, "the video has no sound")
        music = profile.music_ratio or 0.0
        if music < MIN_MUSIC:
            return Applicability(False, f"almost no music ({pct(music)} of the audio)")
        return Applicability(True, f"music in {pct(music)} of the audio")
