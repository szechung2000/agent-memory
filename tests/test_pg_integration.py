"""Integration tests for the pgvector backend (skipped if no PG server)."""

import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("AM_TEST_PG_URI"), reason="no AM_TEST_PG_URI; pg integration skipped"
)


@pytest.fixture()
def pg_repo():
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    from agent_memory.core.models import Memory, MemoryKind
    from agent_memory.db.pg_repository import PgVectorRepository

    uri = os.environ["AM_TEST_PG_URI"]
    if uri.startswith("postgresql://"):
        uri = uri.replace("postgresql://", "postgresql+psycopg://", 1)
    engine = create_engine(uri)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS memories"))
        conn.commit()
    session = sessionmaker(bind=engine)()
    yield PgVectorRepository(session), Memory, MemoryKind
    session.close()
    engine.dispose()


def test_pg_write_and_recall(pg_repo):
    repo, Memory, MemoryKind = pg_repo
    from agent_memory.core.embedding import get_embedder

    emb = get_embedder()
    docs = [
        "pgvector is a postgres extension for vector similarity search",
        "the trading bot uses momentum on TSLA with a stop loss",
        "Sally joined EY in 2006 and passed the CPA soon after",
    ]
    for doc in docs:
        repo.write(Memory(content=doc, kind=MemoryKind.SEMANTIC), emb.embed([doc])[0])

    qvec = emb.embed(["vector similarity extension"])[0]
    results = repo.recall(qvec, "vector similarity extension", k=3)
    assert results[0][0].content.startswith("pgvector")
    assert all(0.0 <= s <= 1.5 for _, s in results)
    # sanity: vectors round-trip through pgvector without shape loss
    row = repo.session.execute(
        __import__("sqlalchemy").text("SELECT dim FROM memories LIMIT 1")
    ).scalar()
    assert int(row) == np.asarray(qvec).size


def test_pg_kind_filter(pg_repo):
    repo, Memory, MemoryKind = pg_repo
    from agent_memory.core.embedding import get_embedder

    emb = get_embedder()
    repo.write(
        Memory(content="episodic alpha event", kind=MemoryKind.EPISODIC),
        emb.embed(["alpha"])[0],
    )
    repo.write(
        Memory(content="semantic beta fact", kind=MemoryKind.SEMANTIC),
        emb.embed(["beta"])[0],
    )

    res = repo.recall(emb.embed(["alpha"])[0], "alpha", k=5, kind="episodic")
    assert [m.content for m, _ in res] == ["episodic alpha event"]
