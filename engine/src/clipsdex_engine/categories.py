"""Categories: the built-in starting set, the user's own changes, and helpers for prompts.

Every category has a one-line description because small local models misread bare names
(asked to pick from "business, edit, movie", one filed pricing advice under "edit").

Users can add, rename, describe, merge and delete categories and move videos between them.
A move is stored as a user assignment (which always wins over automatic ones) plus a
correction, which later shows the categorize model how this user sorts videos.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from sqlmodel import Session, col, delete, select, update

from clipsdex_engine import library
from clipsdex_engine.db.models import (
    AssignedBy,
    Category,
    CategoryAssignment,
    Correction,
    CreatedBy,
    Video,
)

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
MAX_NAME = 40
MAX_DESCRIPTION = 200
COLOR = re.compile(r"#[0-9a-fA-F]{6}")


class CategoryError(ValueError):
    """A category change that can't be done. The message is user-facing."""


@dataclass(frozen=True)
class CategoryCount:
    category: Category
    videos: int  # videos whose current category this is
    by_user: int  # of those, placed there by the user


def ensure_default_categories(session: Session) -> None:
    """Create the built-in categories on first use. Never overwrites user changes."""
    if session.exec(select(Category.id).limit(1)).first() is not None:
        return
    for name, description in DEFAULT_CATEGORIES:
        session.add(Category(name=name, description=description, created_by=CreatedBy.SYSTEM))
    session.flush()


def all_categories(session: Session) -> list[Category]:
    return list(session.exec(select(Category).order_by(col(Category.id))))


def find(session: Session, name: str) -> Category | None:
    """Look a category up by name, ignoring case and extra spaces."""
    wanted = " ".join(name.split()).casefold()
    return next((c for c in all_categories(session) if c.name.casefold() == wanted), None)


def get(session: Session, name: str) -> Category:
    found = find(session, name)
    if found is None:
        known = ", ".join(c.name for c in all_categories(session))
        raise CategoryError(f'There is no category "{name}". Categories: {known}')
    return found


def _clean_name(session: Session, name: str, allow: Category | None = None) -> str:
    name = " ".join(name.split())
    if not name:
        raise CategoryError("A category needs a name.")
    if len(name) > MAX_NAME:
        raise CategoryError(f"Category names can be at most {MAX_NAME} characters.")
    existing = find(session, name)
    if existing is not None and existing is not allow:
        raise CategoryError(f'A category called "{existing.name}" already exists.')
    return name


def _clean_description(description: str | None) -> str | None:
    if description is None:
        return None
    description = " ".join(description.split())
    if len(description) > MAX_DESCRIPTION:
        raise CategoryError(f"Descriptions can be at most {MAX_DESCRIPTION} characters.")
    return description or None


def _clean_color(color: str | None) -> str | None:
    if color is not None and not COLOR.fullmatch(color):
        raise CategoryError("Colors look like #3a7bd5.")
    return color


def create(
    session: Session, name: str, description: str | None = None, color: str | None = None
) -> Category:
    ensure_default_categories(session)
    clean = _clean_name(session, name)
    category = Category(
        name=clean,
        # Without a description the models only see the name, which they often misread.
        description=_clean_description(description) or clean,
        color=_clean_color(color),
        created_by=CreatedBy.USER,
    )
    session.add(category)
    session.flush()
    return category


def rename(session: Session, old: str, new: str) -> Category:
    category = get(session, old)
    category.name = _clean_name(session, new, allow=category)
    session.add(category)
    session.flush()
    return category


def describe(
    session: Session, name: str, description: str | None = None, color: str | None = None
) -> Category:
    """Change the description the models see, and/or the color."""
    category = get(session, name)
    if description is not None:
        category.description = _clean_description(description) or category.name
    if color is not None:
        category.color = _clean_color(color)
    session.add(category)
    session.flush()
    return category


def merge(session: Session, source: str, target: str) -> int:
    """Move everything in ``source`` into ``target`` and delete ``source``.

    Returns how many videos had ``source`` as their current category. History is kept:
    earlier assignments and corrections now point to ``target``.
    """
    src, dst = get(session, source), get(session, target)
    if src.id == dst.id:
        raise CategoryError("A category can't be merged into itself.")
    moved = next(c.videos for c in counts(session) if c.category.id == src.id)
    session.exec(
        update(CategoryAssignment)
        .where(col(CategoryAssignment.category_id) == src.id)
        .values(category_id=dst.id)
    )
    session.exec(
        update(Correction)
        .where(col(Correction.from_category_id) == src.id)
        .values(from_category_id=dst.id)
    )
    session.exec(
        update(Correction)
        .where(col(Correction.to_category_id) == src.id)
        .values(to_category_id=dst.id)
    )
    # A correction from a category to itself teaches nothing.
    session.exec(
        delete(Correction).where(col(Correction.from_category_id) == col(Correction.to_category_id))
    )
    session.delete(src)
    session.flush()
    return moved


def remove(session: Session, name: str) -> None:
    """Delete a category no video was ever put in (otherwise merge it, to keep history)."""
    category = get(session, name)
    used = session.exec(
        select(CategoryAssignment.id).where(CategoryAssignment.category_id == category.id)
    ).first()
    if used is not None:
        raise CategoryError(f'"{category.name}" has videos. Merge it into another category.')
    session.delete(category)
    session.flush()


def move(session: Session, video_ids: Iterable[int], to: str) -> list[int]:
    """Put videos in a category as the user's choice; returns the IDs that changed.

    A change away from another category is also recorded as a correction.
    """
    category = get(session, to)
    assert category.id is not None
    changed: list[int] = []
    for video_id in dict.fromkeys(video_ids):
        if session.get(Video, video_id) is None:
            raise CategoryError(f"There is no video #{video_id}.")
        current, by = library.current_category(session, video_id)
        if current is not None and current.id == category.id and by is AssignedBy.USER:
            continue  # already the user's own choice
        session.add(
            CategoryAssignment(
                video_id=video_id, category_id=category.id, assigned_by=AssignedBy.USER
            )
        )
        if current is None or current.id != category.id:
            session.add(
                Correction(
                    video_id=video_id,
                    from_category_id=current.id if current is not None else None,
                    to_category_id=category.id,
                )
            )
        changed.append(video_id)
    session.flush()
    return changed


def counts(session: Session) -> list[CategoryCount]:
    """Every category with how many videos are in it now."""
    videos: Counter[int] = Counter()
    by_user: Counter[int] = Counter()
    for video_id in session.exec(select(Video.id)):
        assert video_id is not None
        current, by = library.current_category(session, video_id)
        if current is not None and current.id is not None:
            videos[current.id] += 1
            by_user[current.id] += by is AssignedBy.USER
    return [
        CategoryCount(c, videos[c.id or 0], by_user[c.id or 0]) for c in all_categories(session)
    ]
