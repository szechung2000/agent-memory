"""Diagnose a golden case: where do required memories enter the candidate set?

Usage:
    uv run python -m evals.diagnose_case papers p-L3-18 [--k 5] [--mode decompose]

Shows per-sub-query rankings, which required memories surface, and whether the
bottleneck is candidate-set entry (decomposition quality) or final ranking.
"""

from __future__ import annotations

import argparse
import sys

from agent_memory.core.embedding import get_embedder
from agent_memory.core.query_decompose import auto_aspects
from agent_memory.evals.run_golden import load_suite, make_eval_repo


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite")
    parser.add_argument("case_id")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    suite = load_suite(args.suite)
    case = next((c for c in suite["cases"] if c["id"] == args.case_id), None)
    if case is None:
        print(f"case {args.case_id} not found", file=sys.stderr)
        return 1

    emb = get_embedder()
    repo = make_eval_repo()
    from agent_memory.core.models import Memory, MemoryKind

    all_facts = []
    for c in suite["cases"]:
        all_facts.extend(c.get("facts") or c.get("glossary") or [])
    for f in dict.fromkeys(all_facts):
        repo.write(Memory(content=f, kind=MemoryKind.SEMANTIC, namespace="eval"), emb.embed([f])[0])

    q = case["question"]
    aspects = auto_aspects(q)
    required = case.get("required_memories", [])
    print(f"case: {case['id']}  k={args.k}")
    print(f"question: {q}")
    print(f"auto aspects: {aspects}")
    print(f"required: {required}")

    def has(t, spec):
        return any(n.lower() in t.lower() for n in spec.split(" OR "))

    queries = [q] + aspects
    best_rank = {}
    for qi, subq in enumerate(queries):
        results = repo.recall(emb.embed([subq])[0], subq, k=10)
        print(f"\n-- sub-query {qi}: {subq[:75]}")
        shown = 0
        for rank, (m, s) in enumerate(results, 1):
            marks = [r[:40] for r in required if has(m.content, r)]
            if marks or rank <= 3:
                tag = f" <<< REQUIRED {marks}" if marks else ""
                print(f"   #{rank} {s:.3f} | {m.content[:68]}{tag}")
                shown += 1
            for r in required:
                key = r.split(' OR ')[0]
                prev = best_rank.get(key, 99)
                best_rank[key] = min(prev, rank)
        if shown == 0:
            print("   (no relevant hits)")

    print("\n=== diagnosis ===")
    for r in required:
        key = r.split(" OR ")[0]
        rank = best_rank.get(key)
        if rank is None:
            print(f"  '{key}': NEVER RETRIEVED -> decomposition never pointed at this topic")
        elif rank <= args.k:
            print(f"  '{key}': retrieved at #{rank} -> within window, OK")
        else:
            msg = f"  '{key}': retrieved at #{rank} but outside top-{args.k}"
            print(msg + " -> ranking/quota problem")
    return 0


if __name__ == "__main__":
    sys.exit(main())
