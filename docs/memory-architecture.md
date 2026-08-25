# Agent Memory System — Architecture Sketch

Personal knowledgebase + agent memory substrate for Simon's projects
(agentic OS / Telegram bot, trading bot). One storage layer, three consumers:
second brain UI, agentic OS supervisor/worker agents, trading bot agent.

---

## 1. High-level architecture

```
                ┌──────────────────────────────────────┐
   Ingestion    │            MEMORY SERVICE            │   Consumers
                │                                      │
 Notes ───────► │  ┌───────────┐      ┌─────────────┐  │ ───► Second brain (search/browse)
 Docs (PDF/md)──┼─►│ Ingest    │─────►│ Retrieval   │──┼───► Agentic OS agents
 Web clippings ► │  Pipeline  │      │ API (REST)  │  │ ───► Trading bot agent
 Chats ────────► │  └───────────┘      └─────────────┘  │
 Agent state ──► │         │              ▲             │
                 │  Embedding model   Reranker        │
                 └─────────┼──────────────┼────────────┘
                           ▼              │
                 ┌──────────────────────────────┐
                 │ Postgres + pgvector          │
                 │  - semantic memory tables    │
                 │  - episodic memory tables    │
                 │  - entity/knowledge graph    │
                 │ Redis (working memory cache) │
                 └──────────────────────────────┘
```

**Core principle:** one retrieval API, two memory types with different write paths.
Everything is a "memory" record with a type; consumers filter by type and scope.

---

## 2. Memory types

| Type | What | Write path | Retention |
|------|------|-----------|-----------|
| **Semantic** | Facts, notes, docs, market research | Explicit ingestion + consolidation job | Permanent, versioned |
| **Episodic** | Conversation turns, agent events, trade decisions w/ context | Auto-appended during sessions | TTL → summarize → promote to semantic |
| **Working** | Current task state, scratchpad, open tool results | Per-session, hot | Session-scoped (Redis), discarded |
| **Procedural** (later) | Learned workflows, prompts that worked | Distilled from repeated episodes | Permanent |

Key flow: **episodic → semantic promotion**. A nightly consolidation job summarizes
raw episodes into durable facts ("Simon prefers X", "position thesis for TSLA was Y")
and writes them as semantic memories with provenance links.

---

## 3. Schema (Postgres + pgvector)

```sql
-- Core table for all memory records
CREATE TABLE memories (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kind          TEXT NOT NULL CHECK (kind IN ('semantic','episodic','procedural')),
    -- namespacing: which consumer owns/reads this
    namespace     TEXT NOT NULL,            -- e.g. 'brain', 'os', 'trading'
    user_id       TEXT NOT NULL,
    agent_id      TEXT,                     -- nullable; set when an agent wrote it
    title         TEXT,
    content       TEXT NOT NULL,            -- canonical text
    summary       TEXT,                     -- short LLM-generated summary for cheap retrieval
    embedding     VECTOR(1536),             -- OpenAI text-embedding-3-small or similar
    metadata      JSONB NOT NULL DEFAULT '{}',
    -- episodic-specific
    session_id    TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ,              -- TTL for raw episodes
    promoted_from UUID REFERENCES memories(id),
    source        JSONB                     -- provenance: url, file path, chat id
);

CREATE INDEX ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON memories (kind, namespace, user_id);
CREATE INDEX ON memories USING gin (metadata jsonb_path_ops);

-- Entity graph for structured relations (optional v2)
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    namespace TEXT NOT NULL,
    name TEXT NOT NULL,           -- e.g. 'TSLA', 'CS-6515', 'pgvector'
    etype TEXT,                   -- ticker/course/tool/person
    meta JSONB DEFAULT '{}'
);
CREATE TABLE edges (
    src UUID REFERENCES entities(id),
    dst UUID REFERENCES entities(id),
    relation TEXT,                -- 'mentions', 'caused_by', 'supersedes'
    weight REAL DEFAULT 1.0,
    PRIMARY KEY (src, dst, relation)
);
```

`metadata` examples:
- semantic note: `{"tags": ["rl","cs6601"], "doc_type": "lecture_note"}`
- trade episode: `{"ticker":"TSLA","action":"SELL","pnl":-120,"thesis_id":"..."}`

---

## 4. Retrieval strategy — hybrid, always

Single-vector search is not enough. Retrieval = fusion of:

1. **Dense**: pgvector cosine over `embedding` (semantic similarity)
2. **Sparse**: Postgres full-text search (`tsvector`) on content+title (exact terms, tickers)
3. **Structured filters**: SQL on `kind`, `namespace`, `metadata` (time range, ticker)
4. **Recency boost**: score *= decay(created_at) for episodic queries
5. **Rerank**: cross-encoder reranker on top-k fused candidates (e.g. bge-reranker-v2-m3 via a small endpoint) — only for top ~20, keeps latency fine

```
GET /recall?q=...&kind=&namespace=&k=10&filters={...}
→ {memories: [...], scores: {...}}
```

Also expose `GET /context?session=X` — assembles the working-memory prompt block
(recent episodes + pinned facts + relevant semantic hits) so agents make ONE call
to get their context window filled.

---

## 5. Write paths

- **Ingest pipeline** (semantic): chunk (400–800 tok, overlap) → embed batch → upsert with dedupe (content hash + near-dup via cosine > 0.97).
- **Episode append** (episodic): every agent turn appends raw turn + tool results; cheap, no rerank.
- **Consolidation** (nightly): expire stale episodes → LLM-summarize surviving clusters → insert semantic memories with `promoted_from`; never delete originals before promotion succeeds.
- **Conflict handling**: on near-dup insert, keep both but add `supersedes` edge; retrieval prefers newest unless query asks for history (important for "why did I change my mind about X").

---

## 6. Tech stack (fits Python/TS skills)

| Layer | Choice |
|-------|--------|
| Service | FastAPI (Python) |
| DB | Postgres 16 + pgvector (+ Redis for working memory) |
| Embeddings | text-embedding-3-small (cheap) or local bge-m3 later |
| Reranker | bge-reranker-v2-m3 (local, CPU-ok for personal scale) |
| TS client | thin typed client package used by the OS + bot |
| Eval | small eval set of recall questions; measure hit@k before/after changes |

Run locally via docker-compose; deploy later to a single VPS if needed.

---

## 7. Build order

1. **v0 (weekend)**: schema + FastAPI `/remember`, `/recall`, `/context`. Hybrid dense+FTS, no reranker.
2. **v1**: ingestion CLI (markdown/PDF folders, web clipper bookmarklet), consolidation job.
3. **v2**: agent integration — agentic OS tools `memory_write`, `memory_search`; trading bot logs episodes.
4. **v3**: entity graph, procedural memory distillation, recall evals in CI.

## 8. Interview talking points this design demonstrates

- Episodic→semantic consolidation (the hard problem in agent memory)
- Hybrid retrieval with reranking and recency weighting
- Namespaced multi-tenant memory serving multiple agents safely
- Provenance + conflict/supersession handling
- Eval-driven iteration on recall quality
