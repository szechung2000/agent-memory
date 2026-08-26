# agent-memory

Pluggable **semantic + episodic memory service** for LLM agents. One retrieval API serving a personal knowledgebase, agent orchestration, and domain agents (e.g. trading bots).

![status](https://github.com/szechung2000/agent-memory/actions/workflows/ci.yml/badge.svg)

## What it is

- `/remember` — store semantic notes or episodic events, embedded automatically
- `/recall` — hybrid retrieval: dense vector similarity + keyword overlap, fused and scored
- `/context` — one call to assemble prompt-ready memory for an agent
- Namespaced multi-tenant storage (`namespace`, `user_id`, `kind` filters)

## Architecture

```
 Notes/docs ──►┌──────────────────────────┐
 Web clips ───►│      MEMORY SERVICE      │───► Second brain UI
 Chats ───────►│  FastAPI · /remember     │───► Agentic OS agents
 Agent state ► │           /recall        │───► Trading bot agent
               │           /context       │
               └───────────┬──────────────┘
                           ▼
              Postgres+pgvector (prod) or SQLite (dev)
```

Full design: [docs/memory-architecture.md](docs/memory-architecture.md) ·
Roadmap: [docs/project-breakdown.md](docs/project-breakdown.md)

## Quickstart (dev — zero config)

```bash
uv sync
uv run uvicorn agent_memory.api.main:app --reload
```

Runs on SQLite with a local hash embedder (fully offline). Set `AM_OPENAI_API_KEY` for real embeddings.

```bash
curl -X POST localhost:8000/remember -H 'content-type: application/json' \
  -d '{"content":"pgvector enables similarity search in postgres"}'

curl -X POST localhost:8000/recall -H 'content-type: application/json' \
  -d '{"query":"vector search database","k":5}'
```

## Production stack

```bash
cp .env.example .env   # add OPENAI_API_KEY
docker compose up --build
```

Postgres 16 + pgvector, Redis, API service.

## Configuration (env vars, `AM_` prefix)

| Var | Default | Purpose |
|-----|---------|---------|
| `AM_DATABASE_URL` | `sqlite:///./agent_memory.db` | `postgresql+psycopg://…` for pgvector |
| `AM_OPENAI_API_KEY` | – | enables OpenAI embeddings |
| `AM_EMBEDDING_MODEL` | `text-embedding-3-small` | |

## Development

```bash
uv run pytest      # tests
uv run ruff check . # lint
```

## Agent tools

Any LLM agent can use memory via function-calling:

```python
from agent_memory.tools import MemoryToolExecutor
executor = MemoryToolExecutor(repo, embedder)
result = executor.execute("memory_search", {"query": "trading strategy"})
# executor.tools -> OpenAI-compatible tool schemas (memory_write/search/context)
```

## Telegram demo agent

A memory-backed assistant: pulls relevant context before answering, stores durable facts.

```bash
pip install ".[telegram]"
export TELEGRAM_BOT_TOKEN=... AM_OPENAI_API_KEY=...   # key optional — demo mode works without
uv run python -m agent_memory.demo.telegram_agent
```

Demo mode (no key): say "remember that ..." and it stores the fact; ask back and it injects remembered context.

## TypeScript client

Typed HTTP client for the API in `packages/client-ts` (`@agent-memory/client`).

## Roadmap

See [docs/project-breakdown.md](docs/project-breakdown.md): reranking (E3), eval history in CI (E6), live graph dashboard.
