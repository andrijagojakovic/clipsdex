"""The brief prompts against the real local model. uv run pytest -m integration

Prompt sanity checks on short synthetic transcripts, not a quality measure (that is the
golden-set eval with a manual review, step 2.7).
"""

import pytest

from clipsdex_engine import llm
from clipsdex_engine.analyzers import brief
from clipsdex_engine.config import PRESETS, Preset

MODEL = PRESETS[Preset.BALANCED].text
_provider = llm.default_provider()
pytestmark = pytest.mark.integration

PRICING = (
    "Stop charging by the hour, when I switched from $50 an hour to a fixed package of "
    "$2,000, my income doubled, clients pay for results, not for your time, write one clear "
    "package with a fixed price and a clear outcome."
)
FOLLOW_UP = (
    "Most freelancers lose deals because they never follow up. Send a short follow up two "
    "days after your proposal, and another one a week later. Keep it to three sentences. "
    "80% of my deals closed after the second follow up."
)


@pytest.fixture(autouse=True, scope="module")
def _needs_model() -> None:
    if not _provider.health().ok or not _provider.has_model(MODEL):
        pytest.skip(f"Ollama with {MODEL} is not available")


def notes(transcript: str) -> brief.BriefNotes:
    return llm.ask_structured(
        _provider,
        MODEL,
        brief.notes_messages(transcript, "", None),
        brief.BriefNotes,
        max_tokens=brief.NOTES_MAX_TOKENS,
    )


def test_notes_keep_the_videos_numbers() -> None:
    n = notes(PRICING)
    assert n.has_advice and n.advice
    facts = " ".join(brief.clean_list(n.facts, PRICING))
    assert "50" in facts and "2,000" in facts.replace("2000", "2,000")


def test_a_song_lyric_has_no_advice() -> None:
    # Made-up lyrics, not from a real song.
    n = notes("Ooh the summer lights, dancing through the night, hold me close tonight, ooh.")
    assert not n.has_advice


def test_merge_cites_every_video() -> None:
    sources = []
    for number, text in enumerate((PRICING, FOLLOW_UP), start=1):
        n = notes(text)
        sources.append(brief.Source(number, number, f"video {number}", None, None, n.model_dump()))
    report = brief.merge_briefs(_provider, MODEL, sources)
    cited = {s for t in report.brief.themes for p in t.points for s in p.sources}
    assert cited == {1, 2} and report.dropped_sources == []
    assert len(report.brief.themes) >= 2
