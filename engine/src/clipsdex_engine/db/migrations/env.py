"""Alembic environment.

The app runs migrations through ``clipsdex_engine.db.migrate()``, which passes an open
connection. For development (``uv run alembic revision --autogenerate -m "..."`` from
``engine/``), the URL comes from ``alembic.ini`` or ``CLIPSDEX_DATA_DIR``.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy import Connection
from sqlmodel import SQLModel

from clipsdex_engine.config import data_paths
from clipsdex_engine.db import create_db_engine
from clipsdex_engine.db import models as _models  # noqa: F401  (registers the tables)

config = context.config
target_metadata = SQLModel.metadata


def run_with_connection(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=True,  # SQLite can only alter tables by copying them
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        run_with_connection(connection)
        return

    paths = data_paths().ensure()
    engine = create_db_engine(paths.db)
    with engine.begin() as conn:
        run_with_connection(conn)


if context.is_offline_mode():
    raise SystemExit("Offline migrations are not supported; run them against a database.")
run_migrations_online()
