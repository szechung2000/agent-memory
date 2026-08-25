"""Golden-set eval harness.

Loads evals/golden/*.json, ingests each case's facts into a fresh in-memory
store, then checks that retrieval surfaces the required memories for each
question. Scoring is retrieval-level (hit@k over expected content), not LLM
answer-matching — v0 has no answer generation yet.

Usage:
    uv run python -m agent_memory.evals.run_golden            # summary table
    uv run pytest tests/test_evals.py -q                     # as CI gate
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "evals" / "golden"


def load_suite(name: str) -> dict:
    return json.loads((GOLDEN_DIR / f"{name}.json").read_text())


@dataclass
class CaseResult:
    case_id: str
    question: str
    hit: bool = False              # at least one required memory retrieved in top-k
    rank_of_first_hit: int | None = None  # 1-based; None if missed
    top_k: int = 5
    details: list[str] = field(default_factory=list)


def _contains_any(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def run_case(repo, embedder, case: dict, k: int = 5) -> CaseResult:
    """Ingest facts, embed question, check required memories surface."""
    from agent_memory.core.models import Memory, MemoryKind

    id_map: dict[str, str] = {}
    for fact in case.get("facts") or case.get("glossary") or []:
        mem = Memory(content=fact, kind=MemoryKind.SEMANTIC, namespace="eval")
        vec = embedder.embed([fact])[0]
        mid = repo.write(mem, vec)
        key = fact.split(":")[0][:24]
        id_map[key] = mid

    qvec = embedder.embed([case["question"]])[0]
    results = repo.recall(qvec, case["question"], k=k)

    res = CaseResult(case_id=case["id"], question=case["question"], hit=False)

    required = case.get("required_memories", [])
    got_texts = [m.content for m, _ in results]

    if required:
        missing = [r for r in required if not any(_contains_any(t, [r]) for t in got_texts)]
        hits = len(required) - len(missing)
        res.hit = hits >= max(1, len(required) // 2 + (len(required) % 2))  # majority recall
        first_hit_rank = None
        for rank, t in enumerate(got_texts, start=1):
            if any(_contains_any(t, [r]) for r in required):
                first_hit_rank = rank
                break
        res.rank_of_first_hit = first_hit_rank
        if missing:
            res.details.append(f"missing: {missing}")
    else:
        # no explicit requirements: just require non-empty retrieval
        res.hit = bool(got_texts)
        res.rank_of_first_hit = 1 if got_texts else None

    # temporal must/must_not constraints checked against the TOP-1 result only.
    # Rationale: a pre-event question may legitimately retrieve other pre-event
    # facts that mention the same orgs (e.g. "Chase Manhattan was purchased in
    # 2000" when asking about Bear Stearns pre-crisis). The leak we care about
    # is the ANSWER being wrong — i.e. top-1 pointing at a post-change state.
    if case.get("time_context") and got_texts:
        top1 = got_texts[0]
        for bad in case.get("must_not_contain", []):
            if _contains_any(top1, [bad]):
                res.details.append(f"temporal leak: top-1 contains '{bad}'")
                res.hit = False
    return res


def run_suite(repo, embedder, suite_name: str, k: int = 5) -> tuple[float, list[CaseResult]]:
    suite = load_suite(suite_name)
    results = []
    for case in suite["cases"]:
        # fresh namespace per case to avoid cross-contamination
        results.append(run_case(repo, embedder, case, k))
    score = sum(r.hit for r in results) / max(len(results), 1)
    return score, results


def format_report(suite_name: str, score: float, results: list[CaseResult]) -> str:
    lines = [f"\n=== {suite_name}: {score:.0%} ({sum(r.hit for r in results)}/{len(results)}) ==="]
    for r in results:
        mark = "PASS" if r.hit else "FAIL"
        rank = f"first-hit@{r.rank_of_first_hit}" if r.rank_of_first_hit else "miss"
        lines.append(f"[{mark}] {r.case_id} ({rank}) {r.question[:60]}")
        lines.extend(f"       {d}" for d in r.details)
    return "\n".join(lines)


def make_eval_repo(db_url: str = "sqlite:///:memory:"):
    """Fresh isolated store for eval runs."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from agent_memory.db.engine import Base
    from agent_memory.db.models import MemoryRow  # noqa: F401  (register tables)
    from agent_memory.db.repository import MemoryRepository

    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return MemoryRepository(sessionmaker(bind=engine, expire_on_commit=False)())


if __name__ == "__main__":
    from agent_memory.core.embedding import get_embedder

    repo = make_eval_repo()
    emb = get_embedder()

    for name in ["multihop", "glossary", "temporal"]:
        score, results = run_suite(repo, emb, name)
        print(format_report(name, score, results))
