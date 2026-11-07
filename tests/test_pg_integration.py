"""Integration tests for the pgvector backend (skipped if no PG server)."""

import numpy as np
import pytest


@pytest.fixture()
def pg_repo(isolated_pg_uri):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from agent_memory.core.models import Memory, MemoryKind
    from agent_memory.db.pg_repository import PgVectorRepository

    engine = create_engine(isolated_pg_uri)
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


def test_pg_rejects_silent_metadata_filters(pg_repo):
    import pytest

    repo, _, _ = pg_repo
    with pytest.raises(ValueError, match="not implemented"):
        repo.recall(np.ones(384, dtype=np.float32), "loan", filters={"source": "a.txt"})


def test_postgres_cli_bootstrap_and_scoped_ingestion(isolated_pg_uri):
    from agent_memory.core.embedding import LocalHashEmbedder
    from agent_memory.db.engine import create_repository_for_url
    from agent_memory.ingest.pipeline import Ingester

    repo = create_repository_for_url(isolated_pg_uri)
    ingester = Ingester(repo, LocalHashEmbedder())
    text = "Synthetic CMBS loan A has a September remittance."
    try:
        first = ingester.ingest_text(text, namespace="cmbs", metadata={"source": "a.txt"})
        again = ingester.ingest_text(text, namespace="cmbs", metadata={"source": "a.txt"})
        other = ingester.ingest_text(text, namespace="cmbs", metadata={"source": "b.txt"})
        assert first.chunks_written == 1
        assert again.duplicates_skipped == 1
        assert other.chunks_written == 1
        hits = repo.recall(
            LocalHashEmbedder().embed([text])[0], text, k=5, namespace="cmbs"
        )
        assert {m.metadata["source"] for m, _ in hits} == {"a.txt", "b.txt"}
    finally:
        engine = repo.session.get_bind()
        repo.session.close()
        engine.dispose()
