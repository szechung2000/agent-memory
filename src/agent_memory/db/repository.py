"""Memory repository: CRUD + hybrid retrieval (dense + keyword FTS).

Two backends behind one interface:
- SqliteBackend: dev/test default, brute-force cosine over stored float32 blobs
- PostgresBackend: pgvector HNSW + tsvector full-text (activated by AM_DATABASE_URL)

v0 note: SQLite path implements keyword scoring in Python; the retrieval
interface is identical so swapping backends changes nothing upstream.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from agent_memory.core.models import Memory
from agent_memory.core.temporal import temporal_penalty
from agent_memory.db.models import MemoryRow


def _pack(v: np.ndarray) -> bytes:
    return v.astype(np.float32).tobytes()


def _unpack(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32, count=dim)


class MemoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def write(self, m: Memory, vector: np.ndarray) -> str:
        row = MemoryRow(
            id=str(uuid.uuid4()),
            kind=m.kind.value,
            namespace=m.namespace,
            user_id=m.user_id,
            agent_id=m.agent_id,
            title=m.title,
            content=m.content,
            meta=m.metadata,
            session_id=m.session_id,
            embedding=_pack(vector),
            dim=int(vector.size),
        )
        self.session.add(row)
        self.session.commit()
        return row.id

    def get(self, memory_id: str) -> Memory | None:
        row = self.session.get(MemoryRow, memory_id)
        return _to_model(row) if row else None

    def update_metadata(self, memory_id: str, metadata: dict) -> None:
        row = self.session.get(MemoryRow, memory_id)
        if row:
            row.meta = metadata
            self.session.commit()

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
        stmt = select(MemoryRow)
        if kind:
            stmt = stmt.where(MemoryRow.kind == kind)
        if namespace:
            stmt = stmt.where(MemoryRow.namespace == namespace)
        if user_id:
            stmt = stmt.where(MemoryRow.user_id == user_id)
        rows = list(self.session.scalars(stmt))
        q = query_vec.astype(np.float32)

        scored: list[tuple[float, float, MemoryRow]] = []  # (fused, dense, row)
        ql = set(query_text.lower().split())
        for r in rows:
            v = _unpack(r.embedding, r.dim or 0)
            dnorm = np.linalg.norm(v)
            dense = float(np.dot(q, v) / ((np.linalg.norm(q) * dnorm) or 1e-9))
            toks = set((r.content or "").lower().split()) | set((r.title or "").lower().split())
            overlap = len(ql & toks) / max(len(ql), 1)
            fused = 0.7 * max(dense, 0.0) + 0.3 * overlap
            fused -= temporal_penalty(query_text, r.content)
            scored.append((fused, dense, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[tuple[Memory, float]] = []
        for fused, _dense, r in scored[:k]:
            mem = _to_model(r)
            mem.score = round(fused, 4)
            out.append((mem, fused))
        return out


def _to_model(r: MemoryRow) -> Memory:
    from agent_memory.core.models import MemoryKind

    return Memory(
        id=r.id,
        kind=MemoryKind(r.kind),
        namespace=r.namespace,
        user_id=r.user_id,
        agent_id=r.agent_id,
        title=r.title,
        content=r.content,
        metadata=r.meta or {},
        session_id=r.session_id,
        created_at=r.created_at,
    )
