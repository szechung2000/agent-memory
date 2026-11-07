# CMBS RAG quality benchmark

**Status:** runnable, synthetic, small-scale evaluation. This is not a measure of
production accuracy or financial advice. The corpus contains no borrower data.

## Reproduce the A/B experiment

```bash
uv sync --all-groups
uv run python -m agent_memory.evals.cmbs_benchmark --output evals/cmbs/runs/offline.json
```

The dataset has **32 synthetic CMBS source documents** and **45 questions**:

| Category | Questions | What can go wrong |
|---|---:|---|
| Direct lookup | 9 | Wrong loan or field |
| Multi-document | 9 | Missing one of several required sources |
| Temporal | 9 | Using an outdated status or maturity |
| Unanswerable | 9 | Inventing unavailable data |
| Conflicting/restated | 9 | Treating superseded figures as current |

The questions and labels live in [questions.json](../evals/cmbs/questions.json);
the separately stored documents are in
[corpus.json](../evals/cmbs/corpus.json).
Every answerable question has an expected answer, required key facts, and at
least one valid minimal support set. Some have **alternative valid source sets**
so a legitimate correction notice can replace the original source.

### Leakage controls

1. Load and validate the static corpus **without loading answers into memory**.
2. Embed the same fixed corpus once. Both rankers see **the same 20 retrieved candidates**.
3. The baseline uses the repository's dense/keyword hybrid score. The offline
   challenger applies BM25 only to those 20 candidates; it does not see labels.
4. Only the scorer reads support IDs and expected answer terms.
5. Generate live answers from only the question and retrieved evidence IDs/text.
   The model never receives the expected answer or gold supporting IDs.

No per-question gold facts are injected into the corpus; every case queries the
same immutable source collection. This differs from evaluations that generate
their retrieval queries from gold answers.

## Measured offline retrieval result

On 2026-11-07, using **deterministic 384-d hash embeddings**, 20 candidates,
and a **BM25 candidate reranker**:

| Metric | Baseline | BM25 reranked | Change |
|---|---:|---:|---:|
| Evidence Recall@5 | 77.08% | 84.26% | +7.18 pp |
| Evidence Recall@10 | 84.49% | 95.83% | +11.34 pp |
| Full-support@5 | 69.44% | 75.00% | +5.56 pp |
| Full-support@10 | 77.78% | 91.67% | +13.89 pp |
| MRR@10 | 0.6065 | 0.8057 | +0.1992 |

**Important failure analysis:**

- Conflict cases lost one top-5 full-support hit (100% → 88.89%) under BM25;
  keyword reranking can prefer an earlier, superseded report.
- Temporal MRR@10 slipped (0.6667 → 0.6393), even while temporal Recall@10
  improved. An increased aggregate score does not guarantee every question improves.
- Multi-document full-support@5 remained low (44.44% after reranking).
  Retrieval of every required fact is harder than finding any relevant fact.

These measured results apply **only to this synthetic hash-embedding experiment**.
Hash vectors are a deterministic plumbing baseline, not real semantic embeddings.
There are only 36 answerable retrieval cases; the numbers are not stable estimates
of broad CMBS or production performance. No live LLM correctness score was measured.

## Metric definitions

| Metric | Definition |
|---|---|
| Evidence Recall@K | Mean fraction of gold support IDs retrieved in top K, choosing the best valid minimal support set |
| Full-support@K | Fraction of answerable questions whose entire valid support set appears in top K |
| MRR@10 | Mean reciprocal rank of the first valid supporting document in top 10 |
| Citation quote-anchor rate | Fraction of predicted citations whose quote appears verbatim in the named corpus source |
| Gold citation precision | Fraction of predicted citations both quote-anchored and linked to a valid labeled support document |
| Gold citation recall | Average fraction of gold supporting documents cited for answerable questions |
| Abstention precision | Correct unanswerable abstentions divided by all abstentions |
| Abstention recall | Correct unanswerable abstentions divided by all unanswerable questions |
| Key-fact coverage proxy | Fraction of required literal answer terms found in model output |

Undefined precision values are **null**, not a misleading 0 or 100%.
Unanswerable questions are excluded from retrieval recall and included in
abstention evaluation. The key-fact metric is a strict lexical proxy:
paraphrases may score low, and an incorrect answer may still contain the right
terms. Quote anchoring also **does not verify logical entailment**.

## Score actual answer predictions

The default offline command measures **retrieval only**. Answer/citation and
abstention metrics require a prediction for each of the 45 question IDs, from
an actual model run or an external experiment. Example schema:

```json
[
  {
    "case_id": "direct-01",
    "status": "answered",
    "answer": "Loan A's original principal was $20.0 million. [S1]",
    "citations": [
      {
        "doc_id": "terms-a",
        "quote": "Original principal was $20.0 million, coupon 5.20%"
      }
    ]
  },
  {
    "case_id": "unknown-01",
    "status": "insufficient_evidence",
    "answer": "",
    "citations": []
  }
]
```

This short example is illustrative; the CLI requires a **complete file of 45
predictions** to prevent cherry-picked results:

```bash
uv run python -m agent_memory.evals.cmbs_benchmark \
  --predictions evals/cmbs/runs/predictions.json \
  --output evals/cmbs/runs/scored.json
```

For an **explicit opt-in paid model run**, configure `AM_OPENAI_API_KEY` and run:

```bash
uv run python -m agent_memory.evals.cmbs_benchmark \
  --live --live-mode baseline --embedder auto \
  --output evals/cmbs/runs/live-baseline.json
```

Run again with `--live-mode reranked` for generation from the BM25-reordered
evidence. Each run generates up to 45 OpenAI requests, plus embedding requests
when `--embedder auto` selects OpenAI. The CLI never calls the model by default.
Provider outages raise errors instead of silently becoming quality scores.
The live report includes its predictions and model name. Never use confidential
client or employer material in this public benchmark.

## Other retrieval options

- `--reranker cross-encoder` uses the existing optional cross-encoder model.
  Install `local-embeddings` and its model files first. Missing models cause an
  explicit error, not a silent identity rerank.
- `--embedder auto` uses the configured OpenAI or local BGE model; the first
  local download needs network access. Compare models using separate runs.
- `--candidate-pool 20` changes the number of candidates exposed to each
  reranker (minimum 10, maximum corpus size).

Automated tests validate dataset IDs, label isolation, fair A/B candidates,
alternative support sets, citation errors, false abstentions, and scorer
arithmetic. The CI gate tests **mechanics and regressions**; a paid, stochastic
model evaluation is a separate deliberate experiment.
