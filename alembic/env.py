"""Alembic environment: reads AM_DATABASE_URL via app settings."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from agent_memory.core.config import get_settings
from agent_memory.db import models  # noqa: F401  (register tables)
from agent_memory.db.engine import Base

config = config = context.config  # noqa: PLW0127
fileConfig(config.config_file_name)
target_metadata = Base.metadata


def _url() -> str:
    url = get_settings().database_url
    # normalize bare postgresql:// to the psycopg3 driver
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _url()
    connectable = engine_from_config(
        configuration, prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
