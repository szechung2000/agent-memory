# Agent-Memory Dashboard — Design

Reference: "Context Mesh" graph-engineering dashboard (paper-tone, mono type,
pipeline bar → live graph → diagnostics grid). Adapted to what's *meaningful*
for a memory service whose consumers are agents + a second brain.

## Design principles
1. Every panel answers an operator question ("is recall getting worse?"),
   not just decorative stats.
2. Retrieval quality is the product — eval trends are the hero metric,
   graph is secondary until E-graph lands.
3. Temporal integrity gets first-class treatment (we already have leak detection).

---

## Layout (top → bottom, mirroring the reference)

### 1. Header strip
`AGENT-MEMORY • MEMORY OPS` | NODES 1,204 · EPISODES 8,911 · FACTS 2,340 |
embedder: bge-small-en-v1.5 | store: sqlite|pgvector | uptime

### 2. Ingestion pipeline bar (like CHUNK→EXTRACT→RESOLVE→LINK→PRUNE)
`INGEST → CHUNK → EMBED → DEDUPE → STORE → CONSOLIDATE`
- counts per stage (chunks today, near-dupes skipped, embed cost/tokens)
- **CONSOLIDATE highlighted**: episodes pending promotion → promoted today
- anomaly color (red) on stuck stages (e.g. consolidation queue age)

### 3. Hero panel — Recall quality over time (replaces "live graph" initially)
- The three golden suites (multihop / glossary / temporal) as trend lines per commit/run.
- Current score badges vs thresholds; regression = red flash.
- Once entity graph exists (E-v2): swap/split into a live memory-graph view —
  entities, claims (facts), decisions (trade entries), typed edges
  (`supersedes`, `mentions`, `promoted_from`), reusing the reference aesthetic.

### 4. Diagnostics grid (three columns)

**Left — Retrieval observability**
- Recent `/recall` calls: query snippet, top-1 score, latency p50/p95 sparkline
- Score distribution histogram (flag queries whose top-1 < 0.5 as "weak recalls")
- Kind/namespace filter breakdown (semantic vs episodic hit rates)

**Center — What carries the traffic**
- Which memories actually get retrieved: top entities/topics by retrieval count
- Edge ledger analog: `promoted_from` provenance coverage (% facts with lineage),
  supersession chains (how often newest-wins rewired an answer)
- Dead-end ledger analog: **memories never recalled in N days** (candidate for
  pruning or better linking) + queries returning zero results above threshold

**Right — Temporal & eval forensics**
- Timeline of detected state-change events per entity (acquired/merged/rebranded)
- Temporal-leak incidents from golden runs: query, leaked fact, penalty applied
- Walk vs Flat analog: **hybrid vs dense-only hit@k** ablation chart
  (directly produced by our eval harness)

### 5. Footer tagline
`A PROMOTED FACT BEATS A TOP-K GUESS. MEMORY IS WHAT SURVIVES INTO THE NEXT QUESTION.`

---

## Implementation notes
- Stack: FastAPI serves `/dashboard` static page; charts via lightweight
  vanilla JS (uPlot/Chart.js) to keep deps minimal; dark-on-paper palette
  (#f4f1ea bg, #d65a4d accent) matching the reference.
- New read-only endpoints needed:
  - `GET /stats` — counts, stage counters, embedder/store info
  - `GET /stats/recall-log` — ring buffer of recent retrievals (in-proc)
  - `GET /evals/history` — append-only JSONL of golden-run scores (written by harness)
  - `GET /stats/dead-ends` — unretrieved memories + zero-result queries
- Eval harness change: after each run, append {ts, commit?, suite scores} to
  `evals/history.jsonl` (gitignored data dir in prod).
- Phases:
  - D1: header + pipeline counters + eval history chart (data mostly exists)
  - D2: recall log + weak-recall histogram + dead-ends
  - D3: temporal timeline + hybrid-vs-dense ablation panel
  - D4 (post graph epic): live typed-edge graph view replacing/augmenting hero
