"""Ingestion pipeline: files -> chunks -> embedded memories.

- Supports .md, .txt, .pdf (text-extracted), plus raw text
- Markdown frontmatter becomes metadata (tags, doc_type, ...)
- Content-hash dedupe: re-ingesting unchanged content is a no-op
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from agent_memory.core.chunker import chunk_text
from agent_memory.core.models import Memory, MemoryKind
from agent_memory.db.pg_repository import PgVectorRepository
from agent_memory.db.repository import MemoryRepository

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}


@dataclass
class IngestResult:
    files_seen: int = 0
    chunks_written: int = 0
    duplicates_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Return (metadata dict, remaining text). Minimal YAML: key: value pairs."""
    m = _FM_RE.match(text)
    if not m:
        return {}, text
    meta: dict = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            val = val.strip().strip("'\"")
            # inline list support: [a, b]
            if val.startswith("[") and val.endswith("]"):
                meta[key.strip()] = [v.strip() for v in val[1:-1].split(",")]
            else:
                meta[key.strip()] = val
    return meta, text[m.end():]


def extract_pdf(path: Path) -> str:
    """Best-effort PDF text extraction via pypdf if installed."""
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("PDF support requires pypdf: pip install pypdf") from e
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_file(path: Path) -> tuple[dict, str]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, body = parse_frontmatter(raw)
    if path.suffix.lower() == ".pdf":
        body = extract_pdf(path)
    meta.setdefault("source", str(path))
    meta.setdefault("doc_type", path.suffix.lstrip(".").lower())
    return meta, body


def content_hash(text_content: str) -> str:
    return hashlib.sha256(text_content.encode()).hexdigest()


def get_repo_for_url(db_url: str):
    engine = create_engine(db_url)
    if db_url.startswith("sqlite"):
        from agent_memory.db.engine import Base, make_session_factory

        Base.metadata.create_all(engine)
        session = make_session_factory(engine)()
        return MemoryRepository(session)
    session = sessionmaker(bind=engine)()
    return PgVectorRepository(session)


class Ingester:
    def __init__(self, repo, embedder) -> None:
        self.repo = repo
        self.embedder = embedder
        # hash table lives in the DB for persistence across runs
        self._ensure_hash_table()

    def _ensure_hash_table(self) -> None:
        bind = self.repo.session.get_bind() if hasattr(self.repo, "session") else None
        if bind is None:
            return
        with bind.connect() as conn:
            conn.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS ingest_hashes ("
                    "content_hash VARCHAR(64) PRIMARY KEY, memory_id VARCHAR(36))"
                )
            )
            conn.commit()

    def _seen(self, h: str) -> bool:
        bind = self.repo.session.get_bind()
        with bind.connect() as conn:
            row = conn.execute(
                text("SELECT memory_id FROM ingest_hashes WHERE content_hash = :h"), {"h": h}
            ).first()
        return row is not None

    def _record(self, h: str, mid: str) -> None:
        bind = self.repo.session.get_bind()
        with bind.connect() as conn:
            conn.execute(
                text("INSERT INTO ingest_hashes VALUES (:h, :m) ON CONFLICT DO NOTHING"),
                {"h": h, "m": mid},
            )
            conn.commit()

    def ingest_text(
        self,
        body: str,
        metadata: dict | None = None,
        kind: MemoryKind = MemoryKind.SEMANTIC,
        namespace: str = "brain",
        batch_size: int = 32,
    ) -> IngestResult:
        result = IngestResult()
        meta = metadata or {}
        chunks = chunk_text(body)
        if not chunks:
            return result

        pending: list[tuple[str, dict]] = []
        for ch in chunks:
            ch_meta = {**meta, "chunk_index": ch.index}
            if ch.heading:
                ch_meta["heading"] = ch.heading
            h = content_hash(ch.text)
            if self._seen(h):
                result.duplicates_skipped += 1
                continue
            pending.append((ch.text, {**ch_meta, "_hash": h}))

        for i in range(0, len(pending), batch_size):
            batch = pending[i : i + batch_size]
            vecs = self.embedder.embed([t for t, _ in batch])
            for (t, m), v in zip(batch, vecs, strict=True):
                h = m.pop("_hash")
                mem = Memory(
                    content=t,
                    kind=kind,
                    namespace=namespace,
                    title=m.get("heading"),
                    metadata=m,
                )
                mid = self.repo.write(mem, np.asarray(v, dtype=np.float32))
                self._record(h, mid)
                result.chunks_written += 1
        return result

    def ingest_path(self, path: Path, **kwargs) -> IngestResult:
        result = IngestResult(files_seen=0)
        paths = (
            sorted(
                p for p in path.rglob("*")
                if p.suffix.lower() in SUPPORTED_SUFFIXES and p.is_file()
            )
            if path.is_dir()
            else [path]
        )
        for f in paths:
            try:
                meta, body = load_file(f)
                result.files_seen += 1
                sub = self.ingest_text(body, metadata=meta, **kwargs)
                result.chunks_written += sub.chunks_written
                result.duplicates_skipped += sub.duplicates_skipped
                result.errors.extend(sub.errors)
            except Exception as exc:  # noqa: BLE001 — collect per-file errors, keep going
                result.errors.append(f"{f}: {exc}")
        return result
