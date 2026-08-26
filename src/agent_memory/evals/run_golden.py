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
    latency_ms: float | None = None  # question -> recall wall time
    level: int | None = None          # reasoning depth (papers suite)
    details: list[str] = field(default_factory=list)


def _contains_any(text: str, needles: list[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def _aspect_recall(repo, embedder, case: dict, k: int, expand: bool = False,
                   aspects: list[str] | None = None):
    """Retrieve per-aspect sub-queries (one per required memory) + main query; union.

    Multi-part questions have evidence spread across unrelated facts; a single
    query ranks each fact family separately and truncates at k. Sub-querying
    per aspect surfaces every needed family.

    expand=True additionally runs one expansion hop per aspect sub-query
    (bridging terms from first-pass hits), for evidence that shares no surface
    form with the question.
    """
    import numpy as np

    if aspects is not None:
        if len(aspects) < 2:
            qvec = embedder.embed([case["question"]])[0]
            return repo.recall(qvec, case["question"], k=k)
        subqueries = aspects
    else:
        required = case.get("required_memories", [])
        if len(required) < 2:
            qvec = embedder.embed([case["question"]])[0]
            return repo.recall(qvec, case["question"], k=k)
        subqueries = [r.split(" OR ")[0].strip() for r in required]

    merged: dict[str, tuple] = {}
    queries = [case["question"]] + subqueries
    for qi, q in enumerate(queries):
        quota = k if qi == 0 else max(2, k // 2)
        qvec = np.asarray(embedder.embed([q])[0], dtype=np.float32)
        first_pass = repo.recall(qvec, q, k=quota)
        bonus = 1.0 if qi == 0 else 0.92
        for m, s in first_pass:
            adj = s * bonus
            if m.id not in merged or adj > merged[m.id][1]:
                merged[m.id] = (m, adj)

        if expand and qi > 0:
            # one hop: bridge from aspect sub-query's top hit
            from agent_memory.core.multihop import _tokens as _toks

            qtoks = _toks(q)
            expansion: list[str] = []
            for m, _s in sorted(first_pass, key=lambda x: -x[1])[:2]:
                for tok in sorted(_toks(m.content)):
                    if tok not in qtoks and tok not in expansion:
                        expansion.append(tok)
                    if len(expansion) >= 4:
                        break
                if len(expansion) >= 4:
                    break
            if expansion:
                eq = q + " " + " ".join(expansion)
                evec = np.asarray(embedder.embed([eq])[0], dtype=np.float32)
                for m, s in repo.recall(evec, eq, k=2):
                    adj = s * bonus * 0.9
                    if m.id not in merged or adj > merged[m.id][1]:
                        merged[m.id] = (m, adj)
    return sorted(merged.values(), key=lambda x: -x[1])[:k * 2]


def run_case(repo, embedder, case: dict, k: int = 5, multihop: bool = False,
             aspect: bool = False, expand: bool = False, decompose: bool = False) -> CaseResult:
    """Ingest facts, embed question, check required memories surface."""
    from agent_memory.core.models import Memory, MemoryKind

    id_map: dict[str, str] = {}
    for fact in case.get("facts") or case.get("glossary") or []:
        mem = Memory(content=fact, kind=MemoryKind.SEMANTIC, namespace="eval")
        vec = embedder.embed([fact])[0]
        mid = repo.write(mem, vec)
        key = fact.split(":")[0][:24]
        id_map[key] = mid

    import time as _time

    # Multi-part questions need per-aspect coverage, not just global top-k:
    # each required memory family gets its own retrieval quota that survives merging.
    n_required = len(case.get("required_memories", []))
    coverage_mode = decompose or (aspect and n_required >= 2)
    effective_k = max(k, n_required * 2) if coverage_mode else k


    qvec0 = embedder.embed([case["question"]])[0]
    t0 = _time.perf_counter()
    if decompose:
        from agent_memory.core.query_decompose import auto_aspects

        results = _aspect_recall(
            repo, embedder, case, effective_k, expand=expand,
            aspects=auto_aspects(case["question"]) or None,
        )
    elif aspect:
        # oracle decomposition: aspects from golden labels
        aspects = [r.split(" OR ")[0].strip() for r in case.get("required_memories", [])]
        results = _aspect_recall(
            repo, embedder, case, effective_k, expand=expand, aspects=aspects or None
        )
    elif multihop:
        from agent_memory.core.multihop import multihop_recall

        results = multihop_recall(repo, embedder, case["question"], k=k, hops=2)
    else:
        qvec = embedder.embed([case["question"]])[0]
        results = repo.recall(qvec, case["question"], k=k)
    latency_ms = (_time.perf_counter() - t0) * 1000

    res = CaseResult(
        case_id=case["id"],
        question=case["question"],
        hit=False,
        latency_ms=round(latency_ms, 2),
        level=case.get("level"),
    )

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


def run_suite(
    repo,
    embedder,
    suite_name: str,
    k: int = 5,
    multihop: bool = False,
    aspect: bool = False,
    expand: bool = False,
    decompose: bool = False,
) -> tuple[float, list[CaseResult]]:
    suite = load_suite(suite_name)
    results = []
    for case in suite["cases"]:
        results.append(
            run_case(
                repo, embedder, case, k,
                multihop=multihop, aspect=aspect, expand=expand, decompose=decompose,
            )
        )
    score = sum(r.hit for r in results) / max(len(results), 1)
    return score, results


def format_report(suite_name: str, score: float, results: list[CaseResult]) -> str:
    lats = [r.latency_ms for r in results if r.latency_ms is not None]
    lat_summary = ""
    if lats:
        lats_sorted = sorted(lats)
        p50 = lats_sorted[len(lats_sorted) // 2]
        lat_summary = (
            f" | latency ms: p50={p50:.1f} mean={sum(lats)/len(lats):.1f} "
            f"max={max(lats):.1f}"
        )
    lines = [
        f"\n=== {suite_name}: {score:.0%} ({sum(r.hit for r in results)}/{len(results)}) ==="
        + lat_summary
    ]
    for r in results:
        mark = "PASS" if r.hit else "FAIL"
        rank = f"first-hit@{r.rank_of_first_hit}" if r.rank_of_first_hit else "miss"
        lvl = f" L{r.level}" if r.level is not None else ""
        lat = f" {r.latency_ms:.1f}ms" if r.latency_ms is not None else ""
        lines.append(f"[{mark}] {r.case_id}{lvl} ({rank}){lat} {r.question[:60]}")
        # per-level breakdown
        lines.extend(f"       {d}" for d in r.details)
    by_level: dict[int, list[CaseResult]] = {}
    for r in results:
        if r.level is not None:
            by_level.setdefault(r.level, []).append(r)
    if by_level:
        lines.append("  --- by level ---")
        for lvl in sorted(by_level):
            rs = by_level[lvl]
            hit_rate = sum(x.hit for x in rs) / len(rs)
            ls = [x.latency_ms for x in rs if x.latency_ms is not None]
            lat_s = f" mean={sum(ls)/len(ls):.1f}ms" if ls else ""
            lines.append(f"  L{lvl}: {hit_rate:.0%} ({len(rs)} cases){lat_s}")
    return "\n".join(lines)


HISTORY_FILE = Path(__file__).resolve().parents[3] / "evals" / "history.jsonl"


def append_history(suite_scores: dict[str, float], latencies: dict[str, float | None]) -> None:
    """Append one JSON line per eval run: {ts, scores, latency_ms}."""
    import datetime as _dt

    record = {
        "ts": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "scores": {k: round(v, 4) for k, v in suite_scores.items()},
        "latency_ms_mean": {
            k: (round(v, 2) if v is not None else None) for k, v in latencies.items()
        },
    }
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def read_history(limit: int = 20) -> list[dict]:
    """Read last N runs, oldest first."""
    if not HISTORY_FILE.exists():
        return []
    lines = HISTORY_FILE.read_text().strip().splitlines()
    return [json.loads(line) for line in lines[-limit:]]


def print_trend(limit: int = 10) -> None:
    """Print score trend per suite across recent runs."""
    hist = read_history(limit)
    if not hist:
        print("no eval history yet")
        return
    suites = sorted({s for r in hist for s in r["scores"]})
    print(f"\n=== eval history (last {len(hist)} runs) ===")
    header = "run (ts)              " + "".join(f"{s:>12}" for s in suites)
    print(header)
    for rec in hist:
        row = f"{rec['ts']:<22}" + "".join(
            f"{rec['scores'].get(s, float('nan')):>12.0%}" if s in rec["scores"] else f"{'—':>12}"
            for s in suites
        )
        print(row)


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

    scores: dict[str, float] = {}
    lats: dict[str, float | None] = {}
    for name in ["multihop", "glossary", "temporal", "papers"]:
        score, results = run_suite(repo, emb, name)
        scores[name] = score
        lats[name] = (
            sum(r.latency_ms or 0 for r in results) / len(results) if results else None
        )
        print(format_report(name, score, results))

    append_history(scores, lats)
    print_trend()
