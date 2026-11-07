"""SQLAlchemy engine/session setup."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from agent_memory.core.config import get_settings


class Base(DeclarativeBase):
    pass


def get_engine(url: str | None = None):
    url = url or get_settings().database_url
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def make_session_factory(engine=None):
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)


def initialize_database(engine) -> None:
    """One runtime bootstrap for HTTP, CLI, and local evaluations.

    Versioned migrations are a separate future task; do not run DDL per request.
    """
    from agent_memory.db import models as _models  # noqa: F401 (register ORM tables)

    if engine.dialect.name == "postgresql":
        from agent_memory.db.pg_repository import PgVectorRepository

        with Session(engine) as session:
            PgVectorRepository(session)
    else:
        Base.metadata.create_all(engine)


def create_repository_for_url(url: str):
    """Return an initialized repository for CLIs and disposable evaluations."""
    from agent_memory.db.pg_repository import get_repository

    engine = get_engine(url)
    initialize_database(engine)
    return get_repository(make_session_factory(engine)())
