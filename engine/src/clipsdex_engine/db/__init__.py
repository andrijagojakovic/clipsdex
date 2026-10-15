"""Database access: open (and migrate) the SQLite database, sessions, settings storage."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import sqlite_vec
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, event
from sqlmodel import Session, create_engine, select

from clipsdex_engine.config import DataPaths, Settings, data_paths
from clipsdex_engine.db.models import SettingRow

MIGRATIONS_DIR = Path(__file__).parent / "migrations"
# sqlite-vec table of search vectors. Created at run time with the embedding model's size,
# so migrations and the schema check leave it (and its helper tables) alone.
VEC_TABLE = "video_embeddings"


def include_name(name: str | None, type_: str, _parent_names: Any) -> bool:
    """For Alembic: compare every table except the sqlite-vec ones."""
    return not (type_ == "table" and name is not None and name.startswith(VEC_TABLE))


def _sqlite_url(db_path: Path) -> str:
    return f"sqlite:///{db_path.as_posix()}"


def _on_connect(dbapi_connection: Any, _record: Any) -> None:
    # The vector search functions must exist on every connection.
    dbapi_connection.enable_load_extension(True)
    sqlite_vec.load(dbapi_connection)
    dbapi_connection.enable_load_extension(False)
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def create_db_engine(db_path: Path) -> Engine:
    engine = create_engine(_sqlite_url(db_path))
    event.listen(engine, "connect", _on_connect)
    return engine


def alembic_config() -> Config:
    # No sqlalchemy.url: migrate() passes an open connection instead. (A URL would also
    # break on non-ASCII paths, because configparser treats the %-escapes as interpolation.)
    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS_DIR))
    return cfg


def migrate(engine: Engine) -> None:
    """Upgrade the database to the latest schema. Safe to call on every start."""
    cfg = alembic_config()
    with engine.begin() as connection:
        cfg.attributes["connection"] = connection
        command.upgrade(cfg, "head")


def schema_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def head_revision() -> str | None:
    return ScriptDirectory(str(MIGRATIONS_DIR)).get_current_head()


def open_database(paths: DataPaths | None = None) -> Engine:
    """Create the data dir if needed, open the database and migrate it to the latest schema."""
    paths = (paths or data_paths()).ensure()
    engine = create_db_engine(paths.db)
    migrate(engine)
    return engine


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """A session that commits on success and rolls back on error."""
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def load_settings(session: Session) -> Settings:
    """Read stored settings; missing keys use defaults, unknown keys are ignored."""
    stored = {row.key: row.value for row in session.exec(select(SettingRow))}
    known = {k: v for k, v in stored.items() if k in Settings.model_fields}
    return Settings.model_validate(known)


def save_settings(session: Session, settings: Settings) -> None:
    for key, value in settings.model_dump(mode="json").items():
        session.merge(SettingRow(key=key, value=value))
