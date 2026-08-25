"""pgvector-backed repository.

Activated automatically when AM_DATABASE_URL points at Postgres. Uses the same
MemoryRepository interface as the SQLite backend: write()/get()/recall() with
identical scoring semantics (hybrid dense+keyword, fused 0.7/0.3, temporal
penalty) so consumers and evals are backend-agnostic.

Dense search uses pgvector cosine distance via HNSW index; keyword scoring is
computed in Python over candidate rows (v1 — tsvector pushdown is a later
optimization), keeping score parity between backends.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from agent_memory.core.models import Memory, MemoryKind
from agent_memory.core.temporal import temporal_penalty
from agent_memory.db.repository import _to_model


class PgVectorRepository:
    """Drop-in replacement for MemoryRepository when on Postgres."""

    def __init__(self, session: Session) -> None:
        self.session = session
        session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        # table is created by Alembic migrations or create_all fallback
        session.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id VARCHAR(36) PRIMARY KEY,
                    kind VARCHAR(16) NOT NULL,
                    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
                    user_id VARCHAR(64) NOT NULL DEFAULT 'local',
                    agent_id VARCHAR(64),
                    title TEXT,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}',
                    session_id VARCHAR(64),
                    embedding vector(384),
                    dim INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
        )
        session.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_memories_embedding_hnsw "
                "ON memories USING hnsw (embedding vector_cosine_ops)"
            )
        )
        session.commit()

    def write(self, m: Memory, vector: np.ndarray) -> str:
        mid = str(uuid.uuid4())
        v = "[" + ",".join(f"{x:.7g}" for x in vector.astype(np.float32)) + "]"
        self.session.execute(
            text(
                """
                INSERT INTO memories
                    (id, kind, namespace, user_id, agent_id, title, content,
                     metadata, session_id, embedding, dim)
                VALUES
                    (:id, :kind, :namespace, :user_id, :agent_id, :title, :content,
                     CAST(:metadata AS JSONB), :session_id, CAST(:embedding AS vector), :dim)
                """
            ),
            {
                "id": mid,
                "kind": m.kind.value,
                "namespace": m.namespace,
                "user_id": m.user_id,
                "agent_id": m.agent_id,
                "title": m.title,
                "content": m.content,
                "metadata": __import__("json").dumps(m.metadata),
                "session_id": m.session_id,
                "embedding": v,
                "dim": int(vector.size),
            },
        )
        self.session.commit()
        return mid

    def get(self, memory_id: str) -> Memory | None:
        row = self.session.execute(
            text("SELECT * FROM memories WHERE id = :id"), {"id": memory_id}
        ).mappings().first()
        if not row:
            return None
        return Memory(
            id=row["id"],
            kind=MemoryKind(row["kind"]),
            namespace=row["namespace"],
            user_id=row["user_id"],
            agent_id=row["agent_id"],
            title=row["title"],
            content=row["content"],
            metadata=row["metadata"] or {},
            session_id=row["session_id"],
            created_at=row["created_at"],
        )

    def recall(
        self,
        query_vec: np.ndarray,
        query_text: str,
        k: int = 10,
        kind: str | None = None,
        namespace: str | None = None,
        user_id: str | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[Memory, float]]:
        # fetch candidates via dense HNSW ordering (over-fetch for hybrid rerank)
        fetch = max(k * 5, 50)
        q = "[" + ",".join(f"{x:.7g}" for x in query_vec.astype(np.float32)) + "]"
        sql = text(
            """
            SELECT *, embedding <=> CAST(:q AS vector) AS dist
            FROM memories
            WHERE (CAST(:kind AS VARCHAR) IS NULL OR kind = :kind)
              AND (CAST(:namespace AS VARCHAR) IS NULL OR namespace = :namespace)
              AND (CAST(:user_id AS VARCHAR) IS NULL OR user_id = :user_id)
            ORDER BY embedding <=> CAST(:q AS vector)
            LIMIT :fetch
            """
        )
        rows = self.session.execute(
            sql,
            {"q": q, "kind": kind, "namespace": namespace, "user_id": user_id, "fetch": fetch},
        ).mappings().all()

        ql = set(query_text.lower().split())
        scored = []
        for r in rows:
            dense = 1.0 - float(r["dist"])  # cosine distance -> similarity
            toks = set((r["content"] or "").lower().split())
            toks |= set((r["title"] or "").lower().split())
            overlap = len(ql & toks) / max(len(ql), 1)
            fused = 0.7 * max(dense, 0.0) + 0.3 * overlap
            fused -= temporal_penalty(query_text, r["content"])
            scored.append((fused, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for fused, r in scored[:k]:
            mem = _row_to_model(r)
            mem.score = round(fused, 4)
            out.append((mem, fused))
        return out


def _row_to_model(r: Any) -> Memory:
    return Memory(
        id=r["id"],
        kind=MemoryKind(r["kind"]),
        namespace=r["namespace"],
        user_id=r["user_id"],
        agent_id=r["agent_id"],
        title=r["title"],
        content=r["content"],
        metadata=r["metadata"] or {},
        session_id=r["session_id"],
        created_at=r["created_at"],
    )


def get_repository(session: Session):
    """Factory: pgvector backend on Postgres URLs, SQLite backend otherwise."""
    url = session.get_bind().url.render_as_string(hide_password=False)
    if url.startswith("postgresql"):
        return PgVectorRepository(session)
    from agent_memory.db.repository import MemoryRepository

    return MemoryRepository(session)


# re-export for API layer compatibility
_ = _to_model, select
