"""Reproducible CMBS RAG benchmark; labels never enter retrieval or generation.

Run:
  uv run python -m agent_memory.evals.cmbs_benchmark
  uv run python -m agent_memory.evals.cmbs_benchmark --reranker cross-encoder
  uv run python -m agent_memory.evals.cmbs_benchmark --predictions results.json

The default uses hash embeddings and a BM25 candidate reranker for an offline
plumbing benchmark. Neither the retrieval report nor fixture predictions
measure live LLM answer quality. Explicit --live is required to call a model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent_memory.core.answer import (
    CitationValidationError,
    Evidence,
    OpenAIAnswerGenerator,
    verify_draft,
)
from agent_memory.core.embedding import LocalHashEmbedder, get_embedder
from agent_memory.core.models import Memory, MemoryKind
from agent_memory.evals.run_golden import make_eval_repo

DATA_DIR = Path(__file__).resolve().parents[3] / "evals" / "cmbs"
CATEGORIES = {"direct", "multi_document", "temporal", "unanswerable", "conflict"}
NAMESPACE = "cmbs-benchmark"
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "was", "were",
    "is", "are", "did", "do", "does", "what", "which", "who", "how", "when",
    "with", "its", "from", "after", "before", "as", "at", "by", "had",
}


@dataclass(frozen=True)
class CorpusDoc:
    id: str
    source: str
    content: str


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    question: str
    answerable: bool
    support_ids: tuple[str, ...]
    alternative_support_sets: tuple[tuple[str, ...], ...]
    expected_answer: str | None
    must_include: tuple[str, ...]

    @property
    def acceptable_support_sets(self) -> tuple[tuple[str, ...], ...]:
        if not self.answerable:
            return ()
        return (self.support_ids, *self.alternative_support_sets)

    @property
    def acceptable_source_ids(self) -> set[str]:
        return {doc for group in self.acceptable_support_sets for doc in group}


def load_dataset(directory: Path = DATA_DIR) -> tuple[list[CorpusDoc], list[Question]]:
    """Validate dataset consistency before any embeddings, retrieval, or API calls."""
    corpus = json.loads((directory / "corpus.json").read_text(encoding="utf-8"))
    labels = json.loads((directory / "questions.json").read_text(encoding="utf-8"))
    documents = [CorpusDoc(**d) for d in corpus["documents"]]
    questions = [
        Question(
            id=item["id"],
            category=item["category"],
            question=item["question"],
            answerable=item["answerable"],
            support_ids=tuple(item["support_ids"]),
            alternative_support_sets=tuple(
                tuple(ids) for ids in item.get("alternative_support_sets", [])
            ),
            expected_answer=item["expected_answer"],
            must_include=tuple(item["must_include"]),
        )
        for item in labels["cases"]
    ]
    doc_ids = [d.id for d in documents]
    question_ids = [q.id for q in questions]
    if len(doc_ids) != len(set(doc_ids)) or len(question_ids) != len(set(question_ids)):
        raise ValueError("Document IDs and question IDs must be unique")
    if not documents or not questions:
        raise ValueError("Corpus and labels must both be non-empty")
    known = set(doc_ids)
    for d in documents:
        if not d.id or not d.source or not d.content.strip():
            raise ValueError(f"Incomplete source document: {d.id}")
    for q in questions:
        if q.category not in CATEGORIES or not q.question.strip():
            raise ValueError(f"Invalid question: {q.id}")
        if q.answerable:
            if not q.expected_answer or not q.must_include or not q.support_ids:
                raise ValueError(f"Missing positive labels: {q.id}")
            for support_set in q.acceptable_support_sets:
                if not support_set or not set(support_set) <= known:
                    raise ValueError(f"Unknown or empty support set: {q.id}")
        elif q.support_ids or q.alternative_support_sets or q.expected_answer or q.must_include:
            raise ValueError(f"Unanswerable question has positive labels: {q.id}")
    return documents, questions


def _tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.casefold())
        if token not in STOPWORDS
    ]


def bm25_rerank(question: str, candidates: list[tuple]) -> list[tuple]:
    """BM25 over retrieved candidates; never sees case labels or expected answers."""
    if not candidates:
        return []
    token_lists = [_tokens(m.content + " " + (m.title or "")) for m, _ in candidates]
    terms = set(_tokens(question))
    if not terms:
        return candidates[:]
    n = len(candidates)
    avg_length = max(sum(map(len, token_lists)) / n, 1.0)
    doc_freq = Counter(term for tokens in token_lists for term in set(tokens))
    raw_scores = []
    for tokens in token_lists:
        freq = Counter(tokens)
        score = 0.0
        for term in terms:
            if not freq[term]:
                continue
            idf = math.log1p((n - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denom = freq[term] + 1.5 * (0.25 + 0.75 * len(tokens) / avg_length)
            score += idf * freq[term] * 2.5 / denom
        raw_scores.append(score)
    peak = max(raw_scores) or 1.0
    scored = []
    for i, ((memory, original), bm25) in enumerate(zip(candidates, raw_scores, strict=True)):
        combined = 0.55 * bm25 / peak + 0.45 * max(float(original), 0.0)
        scored.append((combined, i, memory, original))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [(m, original) for _, _, m, original in scored]


def _coverage(ranked_ids: list[str], support_sets: tuple[tuple[str, ...], ...], k: int) -> float:
    if not support_sets:
        return 0.0
    found = set(ranked_ids[:k])
    return max(len(found.intersection(s)) / len(s) for s in support_sets)


def _metrics_for_ranked(ranked_ids: list[str], question: Question) -> dict[str, float]:
    return {
        "evidence_recall_at_5": _coverage(ranked_ids, question.acceptable_support_sets, 5),
        "evidence_recall_at_10": _coverage(ranked_ids, question.acceptable_support_sets, 10),
        "full_support_at_5": float(
            _coverage(ranked_ids, question.acceptable_support_sets, 5) == 1.0
        ),
        "full_support_at_10": float(
            _coverage(ranked_ids, question.acceptable_support_sets, 10) == 1.0
        ),
        "mrr_at_10": next(
            (1.0 / rank for rank, doc in enumerate(ranked_ids[:10], 1)
             if doc in question.acceptable_source_ids),
            0.0,
        ),
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 4) if values else None


def _pct_latency(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(percent * len(ordered)) - 1))
    return round(ordered[index], 2)


class Benchmark:
    """Fixed corpus, identical candidate pool, gold labels used only by scorers."""

    def __init__(self, documents: list[CorpusDoc], embedder, candidate_pool: int = 20):
        if not 10 <= candidate_pool <= len(documents):
            raise ValueError("Candidate pool must be between 10 and the corpus size")
        self.documents = {d.id: d for d in documents}
        self.embedder = embedder
        self.candidate_pool = candidate_pool
        self.repo = make_eval_repo()
        vectors = embedder.embed([doc.content for doc in documents])
        for doc, vec in zip(documents, vectors, strict=True):
            self.repo.write(
                Memory(
                    content=doc.content,
                    kind=MemoryKind.SEMANTIC,
                    namespace=NAMESPACE,
                    metadata={"doc_id": doc.id, "source": doc.source},
                ),
                np.asarray(vec, dtype=np.float32),
            )

    def close(self) -> None:
        self.repo.session.close()

    def rankings(self, question: str, reranker: str) -> tuple[list, list, float, float]:
        t0 = time.perf_counter()
        vec = np.asarray(self.embedder.embed([question])[0], dtype=np.float32)
        candidates = self.repo.recall(
            vec, question, k=self.candidate_pool, namespace=NAMESPACE, user_id="local"
        )
        retrieval_ms = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        if reranker == "bm25":
            reranked = bm25_rerank(question, candidates)
        elif reranker == "cross-encoder":
            from agent_memory.core.rerank import _load_cross_encoder
            from agent_memory.core.rerank import rerank as cross_rerank

            if _load_cross_encoder() is None:
                raise RuntimeError(
                    "Cross-encoder model unavailable. Install local-embeddings and download it."
                )
            reranked = cross_rerank(question, candidates, top_k=len(candidates))
        else:
            raise ValueError(f"Unknown reranker: {reranker}")
        rerank_ms = (time.perf_counter() - t1) * 1000
        return candidates, reranked, retrieval_ms, rerank_ms

    def evaluate_retrieval(self, questions: list[Question], reranker: str) -> dict:
        cases: list[dict] = []
        retrieval_times: list[float] = []
        rerank_times: list[float] = []
        for q in questions:
            baseline, reranked, latency, rerank_latency = self.rankings(q.question, reranker)
            retrieval_times.append(latency)
            rerank_times.append(rerank_latency)
            if not q.answerable:
                continue  # unanswerable cases belong to the generation/abstention evaluation
            base_ids = [m.metadata["doc_id"] for m, _ in baseline]
            reranked_ids = [m.metadata["doc_id"] for m, _ in reranked]
            cases.append(
                {
                    "case_id": q.id,
                    "category": q.category,
                    "baseline": _metrics_for_ranked(base_ids, q),
                    "reranked": _metrics_for_ranked(reranked_ids, q),
                    "baseline_top_10": base_ids[:10],
                    "reranked_top_10": reranked_ids[:10],
                }
            )
        names = list(cases[0]["baseline"]) if cases else []
        variants = {
            variant: {
                key: _mean([c[variant][key] for c in cases])
                for key in names
            }
            for variant in ("baseline", "reranked")
        }
        return {
            "questions": len(questions),
            "answerable_scored": len(cases),
            "candidate_pool": self.candidate_pool,
            "reranker": reranker,
            "metrics": variants,
            "delta_reranked_minus_baseline": {
                key: round(variants["reranked"][key] - variants["baseline"][key], 4)
                for key in names
            },
            "latency_ms": {
                "retrieval_p50": _pct_latency(retrieval_times, 0.5),
                "retrieval_p95": _pct_latency(retrieval_times, 0.95),
                "rerank_p50": _pct_latency(rerank_times, 0.5),
                "rerank_p95": _pct_latency(rerank_times, 0.95),
            },
            "by_category": {
                category: {
                    variant: {
                        key: _mean(
                            [c[variant][key] for c in cases if c["category"] == category]
                        )
                        for key in names
                    }
                    for variant in ("baseline", "reranked")
                }
                for category in sorted(CATEGORIES - {"unanswerable"})
            },
            "cases": cases,
        }

    def generate_predictions(
        self, questions: list[Question], generator, reranker: str, mode: str
    ) -> list[dict]:
        """Optional, paid when using a live generator. No gold labels enter prompts."""
        if mode not in {"baseline", "reranked"}:
            raise ValueError("Prediction mode must be baseline or reranked")
        predictions = []
        for q in questions:
            baseline, reranked, _, _ = self.rankings(q.question, reranker)
            selected = baseline if mode == "baseline" else reranked
            evidence = [
                Evidence(
                    ref=f"S{i}",
                    memory_id=m.id,
                    content=m.content,
                    score=float(score),
                    title=m.title,
                    source=m.metadata.get("source"),
                    metadata=m.metadata,
                )
                for i, (m, score) in enumerate(selected[:5], start=1)
            ]
            try:
                raw = generator.generate(q.question, evidence)
                verified = verify_draft(raw, evidence, reranked=(mode == "reranked"))
                prediction = {
                    "case_id": q.id,
                    "status": verified.status,
                    "answer": verified.answer,
                    "citations": [
                        {"doc_id": cite.metadata["doc_id"], "quote": cite.quote}
                        for cite in verified.citations
                    ],
                }
            except CitationValidationError as exc:
                prediction = {
                    "case_id": q.id,
                    "status": "validation_failed",
                    "answer": "",
                    "citations": [],
                    "validation_error": str(exc),
                }
            predictions.append(prediction)
        return predictions


def score_predictions(
    questions: list[Question],
    documents: list[CorpusDoc],
    predictions: list[dict],
    require_complete: bool = True,
) -> dict[str, Any]:
    """Score model/replay output against HELD-OUT labels; no LLM judge is used.

    Key-fact coverage is an exact-string proxy, not semantic factual accuracy.
    Quote anchoring checks provenance, not logical entailment.
    """
    by_case = {q.id: q for q in questions}
    by_doc = {doc.id: doc for doc in documents}
    ids = [pred.get("case_id") for pred in predictions]
    if len(ids) != len(set(ids)) or not set(ids) <= set(by_case):
        raise ValueError("Predictions contain duplicate or unknown case IDs")
    if require_complete and set(ids) != set(by_case):
        raise ValueError("Predictions must cover every benchmark question")

    tp_abstain = all_abstain = unanswerable_count = false_abstain = 0
    citation_total = anchored_total = gold_total = 0
    gold_recalls: list[float] = []
    fact_coverages: list[float] = []
    cases: list[dict] = []
    for pred in predictions:
        q = by_case[pred["case_id"]]
        status = pred["status"]
        if status not in {"answered", "insufficient_evidence", "validation_failed"}:
            raise ValueError(f"Invalid prediction status: {status}")
        raw_citations = pred.get("citations", [])
        if not isinstance(raw_citations, list):
            raise ValueError("Citations must be a list")
        if status != "answered" and raw_citations:
            raise ValueError("Abstentions and validation failures cannot contain citations")
        all_abstain += int(status == "insufficient_evidence")
        unanswerable_count += int(not q.answerable)
        tp_abstain += int(not q.answerable and status == "insufficient_evidence")
        false_abstain += int(q.answerable and status == "insufficient_evidence")
        anchored_ids: set[str] = set()
        correct_citations = 0
        for cited in raw_citations:
            if not isinstance(cited, dict):
                raise ValueError("Each citation must be an object")
            doc_id = cited.get("doc_id")
            quote = cited.get("quote", "")
            doc = by_doc.get(doc_id)
            anchored = bool(
                doc and isinstance(quote, str) and len(" ".join(quote.split())) >= 15
                and " ".join(quote.split()) in " ".join(doc.content.split())
            )
            citation_total += 1
            anchored_total += int(anchored)
            if anchored:
                anchored_ids.add(doc_id)
            if anchored and doc_id in q.acceptable_source_ids:
                correct_citations += 1
                gold_total += 1

        if q.answerable:
            gold_recall = _coverage(
                list(anchored_ids), q.acceptable_support_sets, len(anchored_ids)
            )
            gold_recalls.append(gold_recall)
            text = pred.get("answer", "") if status == "answered" else ""
            if not isinstance(text, str):
                raise ValueError("Answer must be a string")
            fact_coverage = sum(
                token.casefold() in text.casefold() for token in q.must_include
            ) / len(q.must_include)
            fact_coverages.append(fact_coverage)
        else:
            gold_recall = None
            fact_coverage = None
        cases.append(
            {
                "case_id": q.id,
                "category": q.category,
                "status": status,
                "citation_precision": (
                    correct_citations / len(raw_citations) if raw_citations else None
                ),
                "gold_citation_recall": gold_recall,
                "key_fact_coverage": fact_coverage,
            }
        )

    return {
        "evaluated": len(predictions),
        "total_questions": len(questions),
        "complete": len(predictions) == len(questions),
        "citation_quote_anchor_rate": (
            round(anchored_total / citation_total, 4) if citation_total else None
        ),
        "gold_citation_precision": (
            round(gold_total / citation_total, 4) if citation_total else None
        ),
        "gold_citation_recall": _mean(gold_recalls),
        "key_fact_coverage_proxy": _mean(fact_coverages),
        "abstention_precision": (
            round(tp_abstain / all_abstain, 4) if all_abstain else None
        ),
        "abstention_recall": (
            round(tp_abstain / unanswerable_count, 4) if unanswerable_count else None
        ),
        "false_abstentions": false_abstain,
        "validation_failures": sum(p["status"] == "validation_failed" for p in predictions),
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic CMBS RAG benchmark")
    parser.add_argument("--embedder", choices=["hash", "auto"], default="hash")
    parser.add_argument("--reranker", choices=["bm25", "cross-encoder"], default="bm25")
    parser.add_argument("--candidate-pool", type=int, default=20)
    parser.add_argument("--predictions", type=Path, help="Complete external prediction JSON list")
    parser.add_argument(
        "--live", action="store_true", help="Explicitly call paid OpenAI generation"
    )
    parser.add_argument("--live-mode", choices=["baseline", "reranked"], default="baseline")
    parser.add_argument("--output", type=Path, help="Optional local JSON report path")
    args = parser.parse_args(argv)

    if args.predictions and args.live:
        parser.error("Use either --predictions or --live, not both")
    if args.live:
        from agent_memory.core.config import get_settings

        if not get_settings().openai_api_key:
            parser.error("--live requires AM_OPENAI_API_KEY")
    docs, questions = load_dataset()
    embedder = LocalHashEmbedder() if args.embedder == "hash" else get_embedder()
    benchmark = Benchmark(docs, embedder, candidate_pool=args.candidate_pool)
    try:
        report = {
            "dataset": {
                "synthetic": True,
                "documents": len(docs),
                "questions": len(questions),
                "category_counts": dict(sorted(Counter(q.category for q in questions).items())),
                "labels_used_for_generation": False,
            },
            "embedding": args.embedder,
            "retrieval": benchmark.evaluate_retrieval(questions, args.reranker),
            "answer_evaluation": None,
        }
        if args.predictions:
            predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
            report["answer_evaluation"] = score_predictions(questions, docs, predictions)
            report["answer_evaluation_type"] = "external_predictions"
        elif args.live:
            from agent_memory.core.config import get_settings

            settings = get_settings()
            generator = OpenAIAnswerGenerator(
                api_key=settings.openai_api_key, model=settings.generation_model
            )
            predictions = benchmark.generate_predictions(
                questions, generator, args.reranker, args.live_mode
            )
            report["answer_evaluation"] = score_predictions(questions, docs, predictions)
            report["answer_evaluation_type"] = "live_model"
            report["generation_model"] = settings.generation_model
            report["live_mode"] = args.live_mode
            report["predictions"] = predictions
    finally:
        benchmark.close()
    output = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
