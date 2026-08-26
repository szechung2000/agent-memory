# agent-memory

Pluggable **semantic + episodic memory service** for LLM agents — hybrid retrieval, temporal awareness, episodic→semantic consolidation, and graded evals. One retrieval API serving a personal knowledgebase, agent orchestration, and domain agents (e.g. trading bots).

![status](https://github.com/szechung2000/agent-memory/actions/workflows/ci.yml/badge.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Ideas adopted from research

| Idea | Source | Where it lives here |
|---|---|---|
| Hybrid dense + keyword retrieval, fused scores | Lewis et al., *RAG* (arXiv 2005.11401) | `db/repository.py`, `db/pg_repository.py` |
| Non-parametric memory alongside parametric (LLM) memory | *RAG* — the paper's core premise is this service's reason to exist | whole architecture |
| Agentic multi-tool memory access (`memory_write/search/context`) | Yao et al., *ReAct* (arXiv 2210.03629) | `tools.py`, `demo/telegram_agent.py` |
| Query decomposition for multi-part questions; iterative tool-style retrieval | Lelong et al., *Agentic RAG with Knowledge Graphs* (arXiv 2507.16507) | `core/query_decompose.py`, `core/multihop.py` |
| Low-rank adapters as cheap task-switching for the same backbone | Hu et al., *LoRA* (arXiv 2106.09685) — conceptual cousin: one frozen store + swappable namespaces/embedders instead of per-agent stores | embedder tiers, namespacing |

Empirical ablations of what worked (and what didn't): [docs/ablation-report.md](docs/ablation-report.md).
Eval-driven development throughout — every retrieval change must not regress the golden suites.

## What it does

- `/remember` — store semantic facts or episodic events, embedded automatically
- `/recall` — hybrid retrieval: dense vector similarity + keyword overlap, fused and scored
- `/context` — one call assembling prompt-ready memory for an agent
- **Temporal reasoning** — state-change facts dated after a question's time anchor are down-ranked ("who owned X *before* the acquisition?")
- **Consolidation** — nightly distillation of episodes into durable semantic facts with provenance
- **Graded evals** — golden suites incl. 20 research-paper Q&As at reasoning levels L0–L3, latency-timed

## Run it

```bash
# dev — zero config (SQLite + offline BGE embeddings)
uv sync --extra local-embeddings
uv run uvicorn agent_memory.api.main:app --reload

# prod stack (Postgres 16 + pgvector, Redis)
cp .env.example .env   # add OPENAI_API_KEY if wanted
docker compose up --build
```

## API

```bash
curl -X POST localhost:8000/remember -H 'content-type: application/json' \
  -d '{"content":"pgvector enables similarity search in postgres"}'

curl -X POST localhost:8000/recall -H 'content-type: application/json' \
  -d '{"query":"vector search database","k":5}'

curl "localhost:8000/context?query=trading+strategy&k=3"
```

## CLI showcase

```bash
$ am ingest ./my-notes/                 # markdown/pdf/txt -> chunked, deduped memories
files: 2  chunks written: 3  dupes skipped: 0
$ am ingest ./my-notes/                 # re-run = no-op
files: 2  chunks written: 0  dupes skipped: 3
$ am consolidate --review               # episodes -> durable facts, y/N approval queue
episodes scanned: 3  candidates: 2
promoted: 2  superseded: 0
$ am recall "how do reductions work in complexity theory"
0.517 [NP Completeness] | ...SAT is NP-complete by the Cook-Levin theorem...
$ am eval-history                       # score trends across runs
run (ts)                  glossary    multihop      papers    temporal
2026-08-26T04:04:38+00:00     100%         67%         85%        100%
```

## Showcase the capacity: golden evals

The honest measure of a memory system is retrieval quality. Four golden suites (36 cases) gate CI:

| Suite | What it proves | Score |
|---|---|---|
| multihop | linking multiple memories to answer (Sally/CPA/EY) | 67% |
| glossary | term definitions → direct / context / superseding questions | 100% |
| temporal | pre/post event state ("who owned Bear Stearns **before** 2008?") with leak detection | 100% |
| **papers** | 20 Q&A from real arXiv papers at levels L0–L3 | 85% (90% oracle) |

Run them:

```bash
uv run python -m agent_memory.evals.run_golden   # full report + latency per case
uv run pytest tests/test_evals.py                # CI thresholds
am recall --multihop "..."                       # A/B retrieval strategies yourself
```

Papers suite detail — the L0→L3 gradient shows where retrieval ends and synthesis begins:

| Level | Meaning | Hit rate | Latency |
|---|---|---|---|
| L0 | direct fact lookup | 100% | ~1.5ms |
| L1 | one hop | 100% | ~1.8ms |
| L2 | two-fact chains | 100%* | ~40ms |
| L3 | cross-paper synthesis | 60%* | ~47ms |

\* with aspect decomposition; oracle ceiling 90%. Full experiment matrix in [docs/ablation-report.md](docs/ablation-report.md).

## Agent tools

Any LLM agent can use memory via function-calling:

```python
from agent_memory.tools import MemoryToolExecutor
executor = MemoryToolExecutor(repo, embedder)
executor.execute("memory_search", {"query": "trading strategy"})
# executor.tools -> OpenAI-compatible schemas (memory_write / memory_search / memory_context)
```

## Telegram demo agent

```bash
pip install ".[telegram]"
export TELEGRAM_BOT_TOKEN=... AM_OPENAI_API_KEY=...   # key optional — demo mode works without
uv run python -m agent_memory.demo.telegram_agent
```

Demo mode: say "remember that ..." to store a fact; ask back and it injects remembered context.

## TypeScript client

Typed HTTP client in [`packages/client-ts`](packages/client-ts) (`@agent-memory/client`), strict tsc-clean.

## Configuration (env vars, `AM_` prefix)

| Var | Default | Purpose |
|-----|---------|---------|
| `AM_DATABASE_URL` | `sqlite:///./agent_memory.db` | `postgresql+psycopg://…` for pgvector |
| `AM_OPENAI_API_KEY` | – | OpenAI embeddings + LLM decomposer/summarizer |
| `AM_EMBEDDING_MODEL` | `text-embedding-3-small` | |

Embedder priority: OpenAI → local bge-small-en-v1.5 (offline) → hash fallback.

## Development

```bash
uv run pytest        # 45 tests
uv run ruff check .  # lint
```

## Roadmap

Cross-encoder reranking (E3 remainder), live graph dashboard ([design](docs/dashboard-design.md)), full breakdown in [docs/project-breakdown.md](docs/project-breakdown.md).
