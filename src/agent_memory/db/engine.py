"""SQLAlchemy engine/session setup."""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from agent_memory.core.config import get_settings


class Base(DeclarativeBase):
    pass


def get_engine():
    url = get_settings().database_url
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def make_session_factory(engine=None):
    return sessionmaker(bind=engine or get_engine(), expire_on_commit=False)
