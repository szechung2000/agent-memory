# Ablation Report — Retrieval Variants

Golden-eval ablations. Each run: isolated in-memory store, local BGE embedder
(bge-small-en-v1.5, 384-dim), hybrid fusion (0.7 dense + 0.3 keyword), temporal penalty active.

## Embedder tiers (papers suite, single-hop)

| Embedder | Notes |
|---|---|
| hash (256-d) | deterministic bag-of-chars; smoke-level only — papers ~50%, multihop 3/6 fail |
| **bge-small-en-v1.5** | real semantic embeddings, CPU, offline — adopted default |
| text-embedding-3-small | wired, needs API key; expected ≈ or > bge on L2/L3 |

## Single-hop vs multi-hop (iterative query expansion)

| Suite | single | multihop | Δ latency |
|---|---|---|---|
| papers | 70% → 75%* | 75% | +18ms mean |
| — L2 | 60% | 60% | |
| — L3 | 20% | **40%** | |
| multihop suite | 67% | 67% | flat |
| glossary | 100% | 100% | flat |
| temporal | 100% | 100% | flat |

\* run-to-run variance from grouping thresholds; the stable signal is **L3 20%→40%**
with zero regressions elsewhere.

Multi-hop costs ~10x latency of single recall (~20ms vs ~2ms) because it runs two
recalls plus term extraction. At personal-knowledgebase scale this is negligible.

## What retrieval alone can't close

The remaining L2/L3 misses (p-L2-12/13, p-L3-17/18) require **answer synthesis**:
composing multiple retrieved facts into an explanation. Retrieval surfaces all the
needed evidence (first-hit@1 for most), but scoring is evidence-recall, not answer
generation. Closing that gap = generation layer (LLM composing cited facts), which
is out of scope for the memory service itself.

## Score history

`am eval-history` prints per-suite trends across runs (appended to `evals/history.jsonl`,
gitignored). CI gates minimum thresholds per suite via `tests/test_evals.py`.
