"""Analyzers: expensive steps the user asks for (categorize, brief, movie_id, song_id).

``REGISTRY`` maps each name to its analyzer; ``get()`` looks one up by a name the user
typed.
"""

from __future__ import annotations

from clipsdex_engine.analyzers.base import (
    Analyzer,
    AnalyzerContext,
    AnalyzerNotReadyError,
    AnalyzerResult,
    Applicability,
    save_result,
)
from clipsdex_engine.analyzers.brief import Brief
from clipsdex_engine.analyzers.categorize import Categorize
from clipsdex_engine.analyzers.movie_id import MovieId
from clipsdex_engine.analyzers.song_id import SongId

__all__ = [
    "REGISTRY",
    "Analyzer",
    "AnalyzerContext",
    "AnalyzerNotReadyError",
    "AnalyzerResult",
    "Applicability",
    "UnknownAnalyzerError",
    "get",
    "register",
    "save_result",
]


class UnknownAnalyzerError(ValueError):
    pass


REGISTRY: dict[str, Analyzer] = {}


def register(analyzer: Analyzer) -> Analyzer:
    if analyzer.name in REGISTRY:
        raise ValueError(f"Analyzer {analyzer.name!r} is registered twice")
    REGISTRY[analyzer.name] = analyzer
    return analyzer


def get(name: str) -> Analyzer:
    try:
        return REGISTRY[name]
    except KeyError:
        known = ", ".join(REGISTRY)
        raise UnknownAnalyzerError(f"Unknown analyzer {name!r}. Known: {known}") from None


for _analyzer in (Categorize(), Brief(), MovieId(), SongId()):
    register(_analyzer)
