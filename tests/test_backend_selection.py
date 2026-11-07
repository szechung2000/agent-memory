"""Backend selection must not attempt Postgres DDL on every request."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agent_memory.core.embedding import LocalHashEmbedder
from agent_memory.db.pg_repository import PgVectorRepository, get_repository


def test_pg_repository_is_selected_without_connecting_to_the_server():
    engine = create_engine("postgresql+psycopg://memory:memory@localhost:5432/agent_memory")
    with Session(engine) as session:
        assert isinstance(get_repository(session), PgVectorRepository)
    engine.dispose()


def test_fallback_embedding_matches_pgvector_dimensions():
    assert len(LocalHashEmbedder().embed(["test"])[0]) == 384
