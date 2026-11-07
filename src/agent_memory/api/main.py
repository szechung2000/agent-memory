"""FastAPI application: memory storage, retrieval, and cited RAG answers."""

import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from openai import APIConnectionError, APIError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from agent_memory.core.answer import (
    AnswerGenerator,
    AnswerResult,
    CitationValidationError,
    OpenAIAnswerGenerator,
    RerankerUnavailableError,
    answer_question,
)
from agent_memory.core.config import get_settings
from agent_memory.core.embedding import get_embedder
from agent_memory.core.models import Memory, MemoryKind
from agent_memory.db.engine import Base, get_engine, make_session_factory
from agent_memory.db.pg_repository import PgVectorRepository, get_repository

_factory: sessionmaker | None = None
logger = logging.getLogger("agent_memory.api")
_REQUEST_ID = re.compile(r"^[a-zA-Z0-9._-]{1,64}$")


def _event(name: str, **fields: object) -> None:
    """Emit structured metadata; never log memory text or user queries."""
    logger.info(json.dumps({"event": name, **fields}, sort_keys=True))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _factory
    engine = get_engine()
    if engine.dialect.name == "postgresql":
        # The SQLite ORM schema stores BLOBs; Postgres needs the pgvector schema.
        with Session(engine) as session:
            PgVectorRepository(session)  # create the extension/table/indexes once
    else:
        Base.metadata.create_all(engine)
    _factory = make_session_factory(engine)
    app.state.embedder = get_embedder()
    try:
        yield
    finally:
        _factory = None
        engine.dispose()


app = FastAPI(title="agent-memory", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    supplied = request.headers.get("x-request-id", "")
    request_id = supplied if _REQUEST_ID.fullmatch(supplied) else uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _event("request_unhandled", request_id=request_id, method=request.method)
        raise
    route = request.scope.get("route")
    path_template = getattr(route, "path", "<unmatched>")
    _event(
        "http_request",
        request_id=request_id,
        method=request.method,
        route=path_template,
        status_code=response.status_code,
        duration_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(SQLAlchemyError)
async def database_unavailable(request: Request, exc: SQLAlchemyError):
    # Do not leak SQL statements, credentials, or database connection details.
    logger.error("Database operation failed (request_id=%s)", request.state.request_id)
    return JSONResponse(
        status_code=503,
        content={"detail": "Database is temporarily unavailable"},
    )


@app.exception_handler(APIError)
async def model_unavailable(request: Request, exc: APIError):
    if isinstance(exc, APITimeoutError):
        status, message = 504, "Model provider timed out"
    elif isinstance(exc, (RateLimitError, APIConnectionError)):
        status, message = 503, "Model provider is temporarily unavailable"
    elif isinstance(exc, APIStatusError):
        status, message = 502, "Model provider returned an error"
    else:
        status, message = 502, "Model provider request failed"
    logger.warning(
        "Model operation failed (request_id=%s, error_type=%s)",
        request.state.request_id,
        type(exc).__name__,
    )
    return JSONResponse(status_code=status, content={"detail": message})


def get_db():
    if _factory is None:
        raise RuntimeError("The database session factory is not initialized")
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
    namespace: str
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


class AnswerRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=20)
    namespace: str = "default"
    user_id: str = "local"
    rerank: bool = False


def get_answer_generator() -> AnswerGenerator:
    settings = get_settings()
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=503,
            detail="Set AM_OPENAI_API_KEY to enable LLM-generated answers",
        )
    return OpenAIAnswerGenerator(
        api_key=settings.openai_api_key, model=settings.generation_model
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz(db: Session = Depends(get_db)):
    """Readiness requires a successful database round trip."""
    db.execute(text("SELECT 1"))
    return {"status": "ready"}


@app.post("/remember", response_model=RememberResponse)
async def remember(req: RememberRequest, db: Session = Depends(get_db)):
    emb = app.state.embedder
    vec = (await run_in_threadpool(emb.embed, [req.content]))[0]
    mem = Memory(**req.model_dump())
    memory_id = await run_in_threadpool(get_repository(db).write, mem, vec)
    return RememberResponse(id=memory_id)


@app.put("/v1/memories/{external_id:path}", response_model=RememberResponse)
async def upsert_memory(
    external_id: str, req: RememberRequest, db: Session = Depends(get_db)
):
    """Idempotently write a memory using a caller-controlled source key."""
    emb = app.state.embedder
    vec = (await run_in_threadpool(emb.embed, [req.content]))[0]
    mem = Memory(**req.model_dump())
    memory_id = await run_in_threadpool(get_repository(db).upsert, external_id, mem, vec)
    return RememberResponse(id=memory_id)


@app.post("/recall", response_model=list[MemoryOut])
async def recall(req: RecallRequest, db: Session = Depends(get_db)):
    emb = app.state.embedder
    qvec = (await run_in_threadpool(emb.embed, [req.query]))[0]
    results = await run_in_threadpool(
        get_repository(db).recall,
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
            namespace=m.namespace,
            title=m.title,
            content=m.content,
            metadata=m.metadata,
            created_at=m.created_at,
            score=score,
        )
        for m, score in results
    ]


@app.post("/answer", response_model=AnswerResult)
async def answer(
    req: AnswerRequest,
    request: Request,
    db: Session = Depends(get_db),
    generator: AnswerGenerator = Depends(get_answer_generator),
):
    """Generate only source-referenced claims; reject invalid or missing quotes."""
    try:
        result = await run_in_threadpool(
            answer_question,
            get_repository(db),
            app.state.embedder,
            generator,
            req.question,
            k=req.k,
            namespace=req.namespace,
            user_id=req.user_id,
            rerank=req.rerank,
        )
        usage = getattr(generator, "last_usage", None) or {}
        _event(
            "rag_answer",
            request_id=request.state.request_id,
            model=getattr(generator, "model", "test-or-custom"),
            status=result.status,
            retrieved_count=result.retrieved_count,
            reranked=result.reranked,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )
        return result
    except CitationValidationError as exc:
        # Never send unverified model-generated claims back to the caller.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RerankerUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/context")
async def context(
    session: str | None = Query(default=None),
    query: str | None = Query(default=None),
    k: int = 5,
    db: Session = Depends(get_db),
):
    repo = get_repository(db)
    recent = []
    semantic_hits = []
    if query:
        emb = app.state.embedder
        qvec = (await run_in_threadpool(emb.embed, [query]))[0]
        hits = await run_in_threadpool(repo.recall, qvec, query, k)
        semantic_hits = [
            {
                "id": m.id,
                "content": m.content,
                "score": round(s, 4),
                "source": m.metadata.get("source"),
                "metadata": m.metadata,
            }
            for m, s in hits
        ]
    return {"session": session, "semantic": semantic_hits, "recent": recent}
