# Agent-Memory — Project Breakdown

Epics → Stories → Tasks. Target: solo dev, evenings/weekends.
Ordering note: E1 before E2 (schema first), E3 depends on E1, E4 on E2+E3.

---

## Epic 0 — Repo & Foundations
*Get the skeleton right so every later commit is portfolio-grade.*

**Story 0.1 — Project scaffold**
- [ ] Task: docker-compose.yml (postgres+pgvector, redis, api service)
- [ ] Task: Python package layout (`src/agent_memory/`), pyproject with uv
- [ ] Task: lint/format config (ruff), pre-commit hooks
- [ ] Task: GitHub Actions CI — lint + tests on PR
- [ ] Task: top-level README with architecture diagram (from docs/memory-architecture.md)

**Story 0.2 — Docs as first-class artifact**
- [ ] Task: ADR folder (`docs/adr/`) + ADR-0001 "why pgvector over dedicated vector DB"
- [ ] Task: CONTRIBUTING.md (short), LICENSE (MIT)

---

## Epic 1 — Storage Core
*Schema and data access for all four memory types.*

**Story 1.1 — Schema migrations**
- [ ] Task: Alembic setup against Postgres
- [ ] Task: `memories` table migration (kind, namespace, embedding VECTOR, metadata JSONB, TTL fields, provenance)
- [ ] Task: HNSW index on embedding; composite index (kind, namespace, user_id); GIN on metadata
- [ ] Task: `entities`/`edges` tables (v2 placeholder — migrate early, use later)

**Story 1.2 — Repository layer**
- [ ] Task: `MemoryRepository` CRUD (SQLAlchemy or asyncpg) with namespace scoping enforced at query level
- [ ] Task: near-duplicate detection on insert (cosine > 0.97 → link `supersedes`)
- [ ] Task: unit tests with ephemeral Postgres (testcontainers)

---

## Epic 2 — Write Paths
*Getting content in: notes, docs, chats.*

**Story 2.1 — Embedding pipeline**
- [ ] Task: pluggable embedder interface; text-embedding-3-small implementation (+ cost logging)
- [ ] Task: chunker (400–800 tok, overlap, respects markdown headings)
- [ ] Task: batch embed + upsert with idempotency (content hash)

**Story 2.2 — Ingestion CLI**
- [ ] Task: `am ingest <path>` — markdown folders, plain text, PDF
- [ ] Task: metadata extraction from frontmatter (tags, doc_type)
- [ ] Task: re-ingest = update-in-place (match by source path hash)

**Story 2.3 — Episode append API**
- [ ] Task: `POST /episodes` — raw turn + tool results, session-scoped, no rerank
- [ ] Task: TTL enforcement job (expire stale episodes past `expires_at`)

---

## Epic 3 — Retrieval API
*The `/recall` and `/context` endpoints everything consumes.*

**Story 3.1 — Hybrid search**
- [ ] Task: dense search via pgvector cosine (HNSW)
- [ ] Task: sparse search via tsvector full-text
- [ ] Task: structured filters (kind, namespace, metadata JSONB, time range)
- [ ] Task: Reciprocal Rank Fusion of dense+sparse; recency decay factor for episodic kind

**Story 3.2 — Reranking**
- [ ] Task: bge-reranker-v2-m3 local endpoint (CPU-friendly config)
- [ ] Task: rerank top-20 fused candidates behind a feature flag; latency budget check (<300ms p95)

**Story 3.3 — Context assembly**
- [ ] Task: `GET /context?session=X` — recent episodes + pinned facts + semantic hits in one prompt-ready block
- [ ] Task: token-budget-aware truncation (fit N tokens, prioritize pinned > episodic > semantic)

**Story 3.4 — Public API surface**
- [ ] Task: FastAPI app: `POST /remember`, `GET /recall`, `GET /context`, `POST /episodes`
- [ ] Task: Pydantic schemas + OpenAPI docs polish
- [ ] Task: integration test suite (docker-compose up → hit endpoints)

---

## Epic 4 — Consolidation (Episodic → Semantic)
*The hard problem; the main interview talking point.*

**Story 4.1 — Summarization job**
- [ ] Task: nightly job groups unexpired episodes by session/entity
- [ ] Task: LLM summarization → durable fact candidates with provenance (`promoted_from`)
- [ ] Task: human-in-loop option: candidates land in review queue before promotion (CLI approve/reject)

**Story 4.2 — Conflict & supersession**
- [ ] Task: detect contradictions on promote (near-dup w/ differing content)
- [ ] Task: supersede semantics in retrieval — newest wins by default, history available via flag

---

## Epic 5 — Agent Integration
*Prove it works as an agent's brain — feeds the agentic OS.*

**Story 5.1 — Tool layer**
- [ ] Task: `memory_write`, `memory_search`, `memory_get_context` tool definitions (OpenAI function-calling schema)
- [ ] Task: TS typed client package (`packages/client-ts`) for future OS/bot consumers

**Story 5.2 — Reference agent demo**
- [ ] Task: minimal CLI/Telegram demo agent using the tools ("remember X", "what do I know about Y")
- [ ] Task: 60-second demo script/GIF for README

---

## Epic 6 — Evaluation & Observability
*What separates AI-engineering portfolios from API tinkering.*

**Story 6.1 — Recall eval harness**
- [ ] Task: golden set of ~50 question→expected-memory pairs from your own notes
- [ ] Task: hit@k / MRR scoring script, run in CI nightly
- [ ] Task: ablation report page (docs): dense-only vs hybrid vs hybrid+rerank numbers

**Story 6.2 — Ops basics**
- [ ] Task: structured logging w/ request IDs; per-endpoint latency metrics
- [ ] Task: simple `/healthz`; docker-compose healthchecks

---

## Suggested milestone mapping
| Milestone | Epics | Outcome |
|---|---|---|
| v0 (weekend 1–2) | 0, 1, parts of 3 | schema + `/remember` + basic recall working locally |
| v1 (weeks 3–4) | 2, 3 complete | real ingestion + hybrid retrieval + rerank |
| v2 (weeks 5–6) | 4, 5 | consolidation + agent tools + Telegram demo |
| v3 (ongoing) | 6 | evals, ablations, polish for portfolio |
