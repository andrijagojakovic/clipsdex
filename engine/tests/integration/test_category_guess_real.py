"""The category-guess prompt against the real local model. uv run pytest -m integration

These are prompt sanity checks on clear-cut synthetic facts, not an accuracy measure
(that is the golden-set eval, step 2.7).
"""

import pytest

from clipsdex_engine import categories, llm
from clipsdex_engine.db.models import Category, Profile, Video
from clipsdex_engine.profile import category_guess as cg

MODEL = "gemma3:4b"
_provider = llm.default_provider()
pytestmark = pytest.mark.integration

CATS = [Category(name=n, description=d) for n, d in categories.DEFAULT_CATEGORIES]


@pytest.fixture(autouse=True, scope="module")
def _needs_model() -> None:
    if not _provider.health().ok or not _provider.has_model(MODEL):
        pytest.skip(f"Ollama with {MODEL} is not available")


def ask(video: Video, profile: Profile) -> str:
    answer = llm.ask_structured(
        _provider,
        MODEL,
        cg.build_messages(video, profile, CATS),
        cg.guess_schema([c.name for c in CATS]),
        keep_alive="2m",
    )
    return answer.model_dump()["category"]


def test_talking_head_about_pricing_is_business() -> None:
    video = Video(source_id=1, caption="3 pricing mistakes every freelancer makes")
    profile = Profile(
        video_id=1, duration_s=55, cut_count=2, has_audio=True, speech_ratio=0.95,
        music_ratio=0.05, silence_ratio=0.0, has_text=True, ocr_text="MISTAKE #1: HOURLY RATES",
    )  # fmt: skip
    assert ask(video, profile) == "business"


def test_fast_cuts_with_a_song_is_an_edit() -> None:
    video = Video(source_id=1, caption="peaky blinders edit 🔥 #tommyshelby")
    profile = Profile(
        video_id=1, duration_s=18, cut_count=24, has_audio=True, speech_ratio=0.0,
        music_ratio=1.0, silence_ratio=0.0, has_vocals=True, has_text=False,
    )  # fmt: skip
    assert ask(video, profile) == "edit"


def test_film_scene_with_dialogue_is_movie() -> None:
    video = Video(source_id=1, caption="this scene never gets old")
    profile = Profile(
        video_id=1, duration_s=70, cut_count=9, has_audio=True, speech_ratio=0.8,
        music_ratio=0.2, silence_ratio=0.05, has_text=True, ocr_text="The Dark Knight (2008)",
    )  # fmt: skip
    assert ask(video, profile) == "movie"
