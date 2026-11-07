# Reliability boundary: integration tests and API telemetry

This project demonstrates how to validate a source-grounded AI workflow at its
**real persistence boundary**. It is a portfolio prototype, not a production-ready
multi-tenant service.

## How to run the integration gate

```bash
docker compose -p agent-memory-test --profile test up --build \
  --abort-on-container-exit --exit-code-from integration integration
docker compose -p agent-memory-test --profile test down --volumes --remove-orphans
```

The opt-in `test` profile runs tests against PostgreSQL 16 with pgvector. The
database has no host-mapped port and uses a temporary in-memory data directory;
each test gets a unique SQL schema and drops only that test-owned schema.
The live service's database container and persistent named volume are **not**
dependencies of the integration test service.

The GitHub Actions `postgres-integration` job runs the same command on pull
requests and main. The normal `test` job runs lint, unit tests, and the synthetic
CMBS benchmark smoke check. A local run without `AM_TEST_PG_URI` skips the real
database tests and must not be described as an end-to-end PostgreSQL pass.

## Acceptance scenarios

| Scenario | Verification |
|---|---|
| Database startup | Schema bootstrap creates a real 384-dimensional pgvector column and HNSW index |
| Persistence | API writes a CMBS memory and retrieves its stored metadata |
| Idempotency | Repeated `PUT` with the same source key preserves the memory ID and row count |
| Grounded generation | Scripted answer is validated against a real retrieved quote and source |
| Negative case | No retrieved evidence causes abstention with no citations |
| Failure response | Database/provider errors use safe HTTP status and generic response body |
| Request trace | Safe request ID, route template, latency, status and answer metrics are logged |

These checks **do not** measure semantic accuracy of the deterministic embeddings
or factual accuracy of a real model. For that, see [CMBS benchmark](cmbs-benchmark.md).

## API telemetry contract

The `agent_memory.api` logger emits JSON objects. The HTTP event includes
`event=http_request`, `request_id`, `method`, `route`, `status_code`, and
`duration_ms`. The route is a template, not a raw user-supplied path.
For a successful `/answer` call, `event=rag_answer` adds `model`, `status`,
`retrieved_count`, `reranked`, and model token counts when the provider reports
them. Tokens are **not** a cost estimate without a pricing table.

On errors, `/readyz` and other database-dependent routes respond with 503;
an OpenAI provider timeout responds with 504, connection/rate-limit errors with
503, and other provider errors with 502. The service does not echo secrets,
SQL statements, or upstream response bodies to callers. `/healthz` remains a
process liveness probe independent of the database.

No content of a retrieved memory or question is deliberately logged by these
instrumentation paths. Production operators must still review web-server and
reverse-proxy access logging, tracing, retention, and PII policy separately.

## Deliberate limits and next steps

1. Add authentication, tenant-level access controls, request limits, and secret
   management before any public deployment.
2. Replace schema-on-start with reviewed forward migrations, backups and
   restore tests; introduce database connection limits and staged rollouts.
3. Export structured traces and metrics to OpenTelemetry/Prometheus, configure
   alerts, and test behavior under concurrent load and dependency outages.
4. Record model cost with versioned pricing and run the labeled CMBS benchmark
   against live model outputs with safe, public synthetic content.
