"""FastAPI application: /remember, /recall, /context, /healthz."""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, sessionmaker

from agent_memory.core.embedding import get_embedder
from agent_memory.core.models import Memory, MemoryKind
from agent_memory.db.engine import Base, get_engine, make_session_factory
from agent_memory.db.repository import MemoryRepository

_factory: sessionmaker | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _factory
    engine = get_engine()
    Base.metadata.create_all(engine)
    _factory = make_session_factory(engine)
    app.state.embedder = get_embedder()
    yield


app = FastAPI(title="agent-memory", version="0.1.0", lifespan=lifespan)


def get_db():
    s = _factory()
    try:
        yield s
    finally:
        s.close()


class RememberRequest(BaseModel):
    content: str = Field(min_length=1)
    kind: MemoryKind = MemoryKind.SEMANTIC
    namespace: str = "default"
    user_id: str = "local"
    agent_id: str | None = None
    title: str | None = None
    metadata: dict = Field(default_factory=dict)
    session_id: str | None = None


class RememberResponse(BaseModel):
    id: str


class MemoryOut(BaseModel):
    id: str | None
    kind: MemoryKind
    title: str | None
    content: str
    metadata: dict
    created_at: object | None = None
    score: float | None = None


class RecallRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = 10
    kind: MemoryKind | None = None
    namespace: str | None = None
    user_id: str | None = None


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/remember", response_model=RememberResponse)
async def remember(req: RememberRequest, db: Session = Depends(get_db)):
    emb = app.state.embedder
    vec = (await run_in_threadpool(emb.embed, [req.content]))[0]
    mem = Memory(**req.model_dump())
    memory_id = await run_in_threadpool(MemoryRepository(db).write, mem, vec)
    return RememberResponse(id=memory_id)


@app.post("/recall", response_model=list[MemoryOut])
async def recall(req: RecallRequest, db: Session = Depends(get_db)):
    emb = app.state.embedder
    qvec = (await run_in_threadpool(emb.embed, [req.query]))[0]
    results = await run_in_threadpool(
        MemoryRepository(db).recall,
        qvec,
        req.query,
        req.k,
        req.kind.value if req.kind else None,
        req.namespace,
        req.user_id,
    )
    return [
        MemoryOut(
            id=m.id,
            kind=m.kind,
            title=m.title,
            content=m.content,
            metadata=m.metadata,
            created_at=m.created_at,
            score=score,
        )
        for m, score in results
    ]


@app.get("/context")
async def context(
    session: str | None = Query(default=None),
    query: str | None = Query(default=None),
    k: int = 5,
    db: Session = Depends(get_db),
):
    repo = MemoryRepository(db)
    recent = []
    semantic_hits = []
    if query:
        emb = app.state.embedder
        qvec = (await run_in_threadpool(emb.embed, [query]))[0]
        hits = await run_in_threadpool(repo.recall, qvec, query, k)
        semantic_hits = [{"content": m.content, "score": round(s, 4)} for m, s in hits]
    return {"session": session, "semantic": semantic_hits, "recent": recent}
