"""Categories: the built-in starting set, and helpers to read them for prompts.

Users can rename, add and merge categories later (step 5.2). Every category has a one-line
description because small local models misread bare names (CLAUDE.md §15).
"""

from __future__ import annotations

from sqlmodel import Session, col, select

from clipsdex_engine.db.models import Category, CreatedBy

DEFAULT_CATEGORIES: tuple[tuple[str, str], ...] = (
    (
        "edit",
        "A music edit: many short clips cut to a song, often of a movie, show, athlete or "
        "celebrity. Mostly music, little talking.",
    ),
    (
        "business",
        "Advice about work, money, entrepreneurship, marketing, sales, careers or productivity.",
    ),
    ("movie", "A scene or clip from a film or TV series, not set to a different song."),
    (
        "education",
        "Teaches or explains something that is not business: science, history, language, "
        "tech, how-to.",
    ),
    ("fitness", "Workouts, sports training, health or nutrition."),
    ("food", "Recipes, cooking, restaurants or food reviews."),
    ("comedy", "Jokes, sketches, memes or funny moments."),
    ("music", "A song performance, cover, music production or recommendation (not an edit)."),
    ("travel", "Places, trips or travel tips."),
    ("other", "Anything that fits none of the other categories."),
)


def ensure_default_categories(session: Session) -> None:
    """Create the built-in categories on first use. Never overwrites user changes."""
    if session.exec(select(Category.id).limit(1)).first() is not None:
        return
    for name, description in DEFAULT_CATEGORIES:
        session.add(Category(name=name, description=description, created_by=CreatedBy.SYSTEM))
    session.flush()


def all_categories(session: Session) -> list[Category]:
    return list(session.exec(select(Category).order_by(col(Category.id))))
