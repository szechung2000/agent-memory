"""ORM models.

Vector similarity is handled outside these tables:
- Postgres backend: pgvector HNSW index on memories.embedding
- SQLite backend: embeddings stored as BLOB, brute-force cosine in Python
"""

from datetime import datetime

from sqlalchemy import BLOB, JSON, DateTime, Float, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from agent_memory.db.engine import Base


class MemoryRow(Base):
    __tablename__ = "memories"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # semantic|episodic
    namespace: Mapped[str] = mapped_column(String(64), nullable=False, default="default")
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default="local")
    agent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding: Mapped[bytes] = mapped_column(BLOB, nullable=False)
    dim: Mapped[int] = mapped_column(nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_memories_kind_ns_user", "kind", "namespace", "user_id"),
        Index("ix_memories_session", "session_id"),
    )


class EntityRow(Base):
    """v2 placeholder: entity graph nodes."""

    __tablename__ = "entities"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    namespace: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    etype: Mapped[str | None] = mapped_column(String(64), nullable=True)
    meta: Mapped[dict] = mapped_column(JSON, default=dict)


class EdgeRow(Base):
    """v2 placeholder: entity graph edges."""

    __tablename__ = "edges"
    src: Mapped[str] = mapped_column(String(36), primary_key=True)
    dst: Mapped[str] = mapped_column(String(36), primary_key=True)
    relation: Mapped[str] = mapped_column(String(64), primary_key=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
