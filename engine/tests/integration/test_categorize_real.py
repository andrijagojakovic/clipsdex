"""The full categorize prompt against the real local model. uv run pytest -m integration

Prompt sanity checks on clear-cut synthetic inputs, not an accuracy measure (that is the
golden-set eval, step 2.7).
"""

from pathlib import Path
from typing import Any

import pytest

from clipsdex_engine import categories, llm
from clipsdex_engine.analyzers import AnalyzerContext
from clipsdex_engine.analyzers.categorize import Categorize, Example, Prepared
from clipsdex_engine.config import PRESETS, DataPaths, Preset
from clipsdex_engine.db.models import Profile, Video

MODELS = PRESETS[Preset.BALANCED]
_provider = llm.default_provider()
pytestmark = pytest.mark.integration
CATS = list(categories.DEFAULT_CATEGORIES)


@pytest.fixture(autouse=True, scope="module")
def _needs_model() -> None:
    if not _provider.health().ok or not _provider.has_model(MODELS.text):
        pytest.skip(f"Ollama with {MODELS.text} is not available")


def frames(*described: tuple[str, str]) -> dict[str, Any]:
    return {
        "frames": [
            {"t": float(n * 5), "shot_type": shot, "description": text}
            for n, (shot, text) in enumerate(described)
        ]
    }


def categorize(
    inputs: dict[str, Any], examples: list[Example] | None = None, caption: str | None = None
) -> dict[str, Any]:
    ctx = AnalyzerContext(
        video=Video(id=1, source_id=1, caption=caption),
        profile=Profile(video_id=1, duration_s=45, has_audio=True, speech_ratio=0.7,
                        music_ratio=0.3, silence_ratio=0.0, cut_count=6),
        artifacts=inputs,
        models=MODELS,
        paths=DataPaths(Path()),
        provider_factory=lambda: _provider,
        prepared=Prepared(CATS, examples or []),
    )  # fmt: skip
    return Categorize().run(ctx).value


def test_spoken_recipe_without_caption_is_food() -> None:
    value = categorize(
        {
            "transcript": {
                "language": "en",
                "text": "Mix 500 grams of flour with water and your starter, fold it every "
                "30 minutes, then bake at 250 degrees for 40 minutes for a crispy crust.",
            },
            "ocr": {"text": "DAY 3"},
            "frame_descriptions": frames(
                ("food", "Hands knead dough on a floured wooden table."),
                ("food", "A round loaf of bread comes out of an oven."),
            ),
        }
    )
    assert value["category"] == "food"
    assert value["tags"] and value["summary"]


def test_dialogue_and_film_frames_are_a_movie() -> None:
    value = categorize(
        {
            "transcript": {
                "language": "en",
                "text": "You were supposed to be at the bank an hour ago. They know, they "
                "know everything. Get in the car, now.",
            },
            "ocr": {"text": ""},
            "frame_descriptions": frames(
                ("film or TV scene", "Two men in suits argue in a dim warehouse, film look."),
                ("film or TV scene", "A black car speeds down a rainy street at night."),
            ),
        },
        caption="this scene 🔥",
    )
    assert value["category"] == "movie"


def test_follows_the_users_own_choices() -> None:
    examples = [
        Example(
            "Said: <data>Discipline beats motivation. Wake up at 5, do the hard task first, "
            "and your business grows.</data>",
            "business",
            was="fitness",
        ),
        Example(
            "Caption: <data>my 5am routine as a founder</data>\nSummary: A morning routine "
            "with a cold shower and a run before work.",
            "business",
            was="fitness",
        ),
    ]
    value = categorize(
        {
            "transcript": {
                "language": "en",
                "text": "Nobody is coming to save you. Get up early, train, and stop waiting "
                "to feel motivated.",
            },
            "ocr": {"text": "NO EXCUSES"},
            "frame_descriptions": frames(
                ("person talking to camera", "A man in a hoodie talks to the camera at dawn."),
            ),
        },
        examples,
    )
    assert value["category"] == "business"
