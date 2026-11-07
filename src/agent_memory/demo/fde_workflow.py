"""A reproducible, no-API-key example of the agent-memory tool workflow.

All CMBS facts are synthetic. This demonstrates retrieval and context plumbing,
not LLM answer generation or live security/loan analysis.

Run: uv run python -m agent_memory.demo.fde_workflow
"""

from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from agent_memory.core.embedding import LocalHashEmbedder
from agent_memory.ingest.pipeline import Ingester, get_repo_for_url
from agent_memory.tools import MemoryToolExecutor

SAMPLE_DOCS = [
    (
        "synthetic/servicer-remittance-sep-2026.txt",
        "CMBS Loan A: September 2026 remittance reports a 30-day delinquency.",
    ),
    (
        "synthetic/loan-tape-sep-2026.txt",
        "CMBS Loan A: September 2026 scheduled loan balance is $12 million.",
    ),
    (
        "synthetic/asset-manager-notes.txt",
        "CMBS Loan B: September 2026 remittance reports current payment status.",
    ),
]


def run_demo() -> dict:
    """Return a trace of ingest -> retrieval -> context -> episode write."""
    with TemporaryDirectory() as workdir:
        repo = get_repo_for_url(f"sqlite:///{workdir}/demo.db")
        embedder = LocalHashEmbedder()  # deterministic; no network or credentials
        ingester = Ingester(repo, embedder)

        written = 0
        skipped = 0
        for source, content in SAMPLE_DOCS:
            result = ingester.ingest_text(
                content, metadata={"source": source}, namespace="cmbs-demo"
            )
            written += result.chunks_written
            # Repeat the same source to demonstrate an idempotent import.
            skipped += ingester.ingest_text(
                content, metadata={"source": source}, namespace="cmbs-demo"
            ).duplicates_skipped

        executor = MemoryToolExecutor(repo, embedder)
        query = "Which CMBS loan had a September 2026 delinquency?"
        hits = json.loads(
            executor.execute("memory_search", {"query": query, "namespace": "cmbs-demo"})
        )
        context = json.loads(
            executor.execute("memory_context", {"topic": query, "k": 3})
        )
        event = json.loads(
            executor.execute(
                "memory_write",
                {
                    "content": "Analyst requested a September 2026 delinquency review.",
                    "kind": "episodic",
                    "namespace": "cmbs-demo",
                },
            )
        )
        repo.session.close()
        return {
            "synthetic_data": True,
            "ingest": {"chunks_written": written, "duplicates_skipped": skipped},
            "tool_calls": ["memory_search", "memory_context", "memory_write"],
            "query": query,
            "retrieved_evidence": hits,
            "context": context,
            "episode": event,
        }


def main() -> None:
    print(json.dumps(run_demo(), indent=2))


if __name__ == "__main__":
    main()
