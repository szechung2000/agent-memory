"""A/B: single-hop vs multi-hop retrieval per suite."""

from agent_memory.core.embedding import get_embedder
from agent_memory.evals.run_golden import make_eval_repo, run_suite

repo = make_eval_repo()
emb = get_embedder()

for name in ["papers", "multihop", "glossary", "temporal"]:
    s0, r0 = run_suite(make_eval_repo(), emb, name, multihop=False)
    s1, r1 = run_suite(make_eval_repo(), emb, name, multihop=True)
    print(f"{name}: single={s0:.0%} multihop={s1:.0%}")
    lats1 = [x.latency_ms for x in r1 if x.latency_ms is not None]
    if lats1:
        print(f"  multihop latency: mean={sum(lats1)/len(lats1):.1f}ms max={max(lats1):.1f}ms")
