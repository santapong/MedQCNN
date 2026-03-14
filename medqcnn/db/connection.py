"""
Database connection management for MedQCNN.

Supports PostgreSQL (production) and SQLite (dev/testing fallback).
Set DATABASE_URL env var for PostgreSQL, otherwise falls back to SQLite.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from medqcnn.db.models import Base

_DEFAULT_SQLITE = "sqlite:///medqcnn.db"

_engine = None
_SessionFactory: sessionmaker[Session] | None = None


def get_database_url() -> str:
    """Return the database URL from env or fallback to SQLite."""
    return os.environ.get("DATABASE_URL", _DEFAULT_SQLITE)


def get_engine():
    """Get or create the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = get_database_url()
        connect_args = {}
        if url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(url, connect_args=connect_args, echo=False)
    return _engine


def init_db() -> None:
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def get_session() -> Session:
    """Return a new database session."""
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager that initializes the DB and yields a session.

    Usage::

        with db_session() as session:
            create_prediction(session, ...)
    """
    init_db()
    session = get_session()
    try:
        yield session
    finally:
        session.close()


def reset_engine() -> None:
    """Reset the engine (useful for testing)."""
    global _engine, _SessionFactory
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionFactory = None
