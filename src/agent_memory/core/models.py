"""Domain models shared across layers."""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class MemoryKind(StrEnum):
    SEMANTIC = "semantic"
    EPISODIC = "episodic"


class Memory(BaseModel):
    id: str | None = None
    kind: MemoryKind = MemoryKind.SEMANTIC
    namespace: str = "default"
    user_id: str = "local"
    agent_id: str | None = None
    title: str | None = None
    content: str
    metadata: dict = Field(default_factory=dict)
    session_id: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    score: float | None = None  # set on retrieval, not stored
