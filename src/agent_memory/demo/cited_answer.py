"""Offline FDE demo: the real retrieval/verifier with a scripted generator.

No OpenAI request is made. The scripted generator is deliberately visible and
must not be presented as an LLM-generated answer.

Run: uv run python -m agent_memory.demo.cited_answer
"""

from __future__ import annotations

import json
from tempfile import TemporaryDirectory

from agent_memory.core.answer import Evidence, answer_question
from agent_memory.core.embedding import LocalHashEmbedder
from agent_memory.demo.fde_workflow import SAMPLE_DOCS
from agent_memory.ingest.pipeline import Ingester, get_repo_for_url


class ScriptedCMBSGenerator:
    """Return one fixed claim citing the retrieved synthetic remittance."""

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        matching = next((e for e in evidence if "30-day delinquency" in e.content), None)
        if matching is None:
            return json.dumps({"insufficient_evidence": True, "claims": []})
        return json.dumps(
            {
                "insufficient_evidence": False,
                "claims": [
                    {
                        "text": "Synthetic CMBS Loan A was 30 days delinquent in September 2026.",
                        "citations": [
                            {"source_id": matching.ref, "quote": matching.content}
                        ],
                    }
                ],
            }
        )


def run_demo() -> dict:
    with TemporaryDirectory() as workdir:
        repo = get_repo_for_url(f"sqlite:///{workdir}/demo.db")
        embedder = LocalHashEmbedder()
        ingester = Ingester(repo, embedder)
        for source, content in SAMPLE_DOCS:
            ingester.ingest_text(
                content, metadata={"source": source}, namespace="cmbs-demo"
            )
        result = answer_question(
            repo,
            embedder,
            ScriptedCMBSGenerator(),
            "Which CMBS loan was delinquent in September 2026?",
            k=3,
            namespace="cmbs-demo",
        )
        repo.session.close()
        return {
            "mode": "scripted_generator_not_live_llm",
            "verification": "retrieved source IDs and exact text quotes",
            **result.model_dump(),
        }


def main() -> None:
    print(json.dumps(run_demo(), indent=2))


if __name__ == "__main__":
    main()
