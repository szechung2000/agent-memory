"""Read-only, offline CMBS RAG explorer for FDE portfolio walkthroughs.

Run: uv run uvicorn agent_memory.demo.web:app --host 127.0.0.1 --port 8000
No cloud model, external network request, or persistent database is required.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from agent_memory.core.embedding import LocalHashEmbedder
from agent_memory.demo.cited_answer import run_demo
from agent_memory.evals.cmbs_benchmark import (
    Benchmark,
    Question,
    _metrics_for_ranked,
    load_dataset,
)

app = FastAPI(
    title="CMBS RAG Explorer",
    description="Synthetic, offline and reproducible retrieval/citation demonstration",
)

PAGE = Path(__file__).parent / "static" / "index.html"
SCENARIOS = {
    "supported": "Which CMBS loan was delinquent in September 2026?",
    "abstain": "What insurance premium did Loan A pay in September 2026?",
}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(PAGE, media_type="text/html")


@app.get("/api/questions")
def questions():
    _, dataset = load_dataset()
    return {
        "synthetic": True,
        "cases": [
            {"id": q.id, "category": q.category, "question": q.question}
            for q in dataset
        ],
    }


@lru_cache(maxsize=1)
def _benchmark_summary() -> dict:
    docs, questions = load_dataset()
    benchmark = Benchmark(docs, LocalHashEmbedder())
    try:
        report = benchmark.evaluate_retrieval(questions, "bm25")
    finally:
        benchmark.close()
    return {
        "synthetic": True,
        "generation_evaluated": False,
        "documents": len(docs),
        "questions": len(questions),
        "answerable_scored": report["answerable_scored"],
        "reranker": "BM25 over the same dense candidate pool",
        "metrics": report["metrics"],
        "delta": report["delta_reranked_minus_baseline"],
        "latency_ms": report["latency_ms"],
        "by_category": report["by_category"],
    }


@app.get("/api/benchmark")
def benchmark():
    return _benchmark_summary()


def _display_results(ranked: list, gold: set[str]) -> list[dict]:
    return [
        {
            "rank": i,
            "doc_id": m.metadata["doc_id"],
            "source": m.metadata["source"],
            "content": m.content,
            "score": round(float(score), 4),
            "gold_evidence": m.metadata["doc_id"] in gold,
        }
        for i, (m, score) in enumerate(ranked[:5], 1)
    ]


def _variant(ranked: list, question: Question) -> dict:
    ids = [m.metadata["doc_id"] for m, _ in ranked]
    return {
        "top_5": _display_results(ranked, question.acceptable_source_ids),
        "metrics": (
            _metrics_for_ranked(ids, question) if question.answerable else None
        ),
    }


@app.get("/api/questions/{case_id}")
def inspect_case(case_id: str):
    docs, questions = load_dataset()
    question = next((q for q in questions if q.id == case_id), None)
    if question is None:
        raise HTTPException(status_code=404, detail="Unknown synthetic benchmark case")

    benchmark = Benchmark(docs, LocalHashEmbedder())
    try:
        baseline, reranked, retrieval_ms, rerank_ms = benchmark.rankings(
            question.question, "bm25"
        )
    finally:
        benchmark.close()
    return {
        "synthetic": True,
        "case": {
            "id": question.id,
            "category": question.category,
            "question": question.question,
            "answerable": question.answerable,
            "expected_answer": question.expected_answer,
            "gold_support_ids": sorted(question.acceptable_source_ids),
        },
        "baseline": _variant(baseline, question),
        "reranked": _variant(reranked, question),
        "latency_ms": {
            "retrieval": round(retrieval_ms, 2),
            "reranking": round(rerank_ms, 2),
        },
        "note": "Gold answers and evidence IDs are used only for scoring and display; "
        "they are never supplied to retrieval or the generator.",
    }


@app.get("/api/citations/{scenario}")
def cited_demo(scenario: str):
    if scenario not in SCENARIOS:
        raise HTTPException(status_code=404, detail="Unknown scripted demo scenario")
    return {"synthetic": True, "scenario": scenario, **run_demo(SCENARIOS[scenario])}
