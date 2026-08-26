/**
 * Typed TypeScript client for the agent-memory HTTP API.
 *
 * ```ts
 * const mem = new MemoryClient("http://localhost:8000");
 * await mem.remember({ content: "Prefers Python", namespace: "brain" });
 * const hits = await mem.recall({ query: "language preference" });
 * ```
 */

export type MemoryKind = "semantic" | "episodic";

export interface RememberRequest {
  content: string;
  kind?: MemoryKind;
  namespace?: string;
  user_id?: string;
  agent_id?: string | null;
  title?: string | null;
  metadata?: Record<string, unknown>;
  session_id?: string | null;
}

export interface RecallRequest {
  query: string;
  k?: number;
  kind?: MemoryKind | null;
  namespace?: string | null;
  user_id?: string | null;
}

export interface MemoryHit {
  id: string | null;
  kind: MemoryKind;
  title: string | null;
  content: string;
  metadata: Record<string, unknown>;
  created_at: string | null;
  score: number | null;
}

export interface ContextResponse {
  session: string | null;
  semantic: { content: string; score: number }[];
  recent: unknown[];
}

export class MemoryClient {
  constructor(
    private readonly baseUrl: string,
    private readonly fetchImpl: typeof fetch = globalThis.fetch,
  ) {}

  private async post<T>(path: string, body: unknown): Promise<T> {
    const res = await this.fetchImpl(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`agent-memory ${path}: ${res.status} ${await res.text()}`);
    return (await res.json()) as T;
  }

  async get<T>(path: string): Promise<T> {
    const res = await this.fetchImpl(`${this.baseUrl}${path}`);
    if (!res.ok) throw new Error(`agent-memory ${path}: ${res.status} ${await res.text()}`);
    return (await res.json()) as T;
  }

  remember(req: RememberRequest): Promise<{ id: string }> {
    return this.post("/remember", req);
  }

  recall(req: RecallRequest): Promise<MemoryHit[]> {
    return this.post("/recall", req);
  }

  context(topic: string, k = 5, sessionId?: string): Promise<ContextResponse> {
    const params = new URLSearchParams({ query: topic, k: String(k) });
    if (sessionId) params.set("session", sessionId);
    return this.get(`/context?${params.toString()}`);
  }

  health(): Promise<{ status: string }> {
    return this.get("/healthz");
  }
}
