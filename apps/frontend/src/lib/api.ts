export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export interface ModelMetadata {
  id: string;
  name: string;
  provider: string;
  architecture: string;
  context_length: number;
  parameter_count?: string;
  hardware_tier: string;
  is_local: boolean;
  requires_gpu: boolean;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface CandidateTokenData {
  token_id: number;
  token_text: string;
  prob: number;
  logprob: number;
}

export interface TokenTelemetryData {
  index: number;
  token_id: number;
  token_text: string;
  prob: number;
  logprob: number;
  surprisal_bits: number;
  entropy_bits: number;
  latency_ms: number;
  top_k: CandidateTokenData[];
}

export interface SequenceTelemetryData {
  tokens: TokenTelemetryData[];
  text: string;
  total_tokens: number;
  total_duration_ms: number;
  tokens_per_second: number;
  mean_surprisal_bits: number;
  perplexity: number;
  mean_entropy_bits: number;
  max_surprisal_token?: TokenTelemetryData | null;
  min_surprisal_token?: TokenTelemetryData | null;
}

export interface ChatTelemetry {
  ttftMs?: number;
  totalLatencyMs?: number;
  tokensPerSecond?: number;
  totalCostUsd?: number;
  isFree?: boolean;
  tokenSequence?: SequenceTelemetryData;
}

export interface ArenaModelTarget {
  model: string;
  provider?: string;
}

export interface ArenaMetricResult {
  model: string;
  provider: string;
  success: boolean;
  output_text: string;
  ttft_ms: number | null;
  total_latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  tokens_per_second: number;
  cost_usd: number;
  is_free: boolean;
  error: string | null;
}

export interface ArenaComparisonResponse {
  prompt: string;
  total_models: number;
  successful_models: number;
  results: ArenaMetricResult[];
  rankings: {
    fastest_ttft: string | null;
    highest_throughput: string | null;
    lowest_cost: string | null;
  };
}

export interface ModelEloRecord {
  model_id: string;
  rating: number;
  matches: number;
  wins: number;
  losses: number;
  ties: number;
  win_rate: number;
  ci_lower: number;
  ci_upper: number;
  category_ratings: Record<string, number>;
}

export interface JudgeVerdict {
  winner: string;
  score_a: number;
  score_b: number;
  confidence: number;
  rationale: string;
  position_swapped: boolean;
  criteria_breakdown: Record<string, Record<string, number>>;
}

export interface ArenaBattleResponse {
  match: {
    match_id: string;
    model_a: string;
    model_b: string;
    winner: string;
    category: string;
    rating_a_before: number;
    rating_b_before: number;
    rating_a_after: number;
    rating_b_after: number;
    delta_a: number;
    delta_b: number;
  };
  verdict: JudgeVerdict;
  completion_a: string;
  completion_b: string;
}

export async function fetchEloLeaderboard(category = "overall"): Promise<ModelEloRecord[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/arena/leaderboard?category=${encodeURIComponent(category)}`);
    if (!res.ok) throw new Error("Failed to fetch Elo leaderboard");
    return await res.json();
  } catch (err) {
    console.warn("Could not fetch Elo leaderboard:", err);
    return [];
  }
}

export async function runArenaBattle(
  prompt: string,
  modelA: ArenaModelTarget,
  modelB: ArenaModelTarget,
  category = "overall"
): Promise<ArenaBattleResponse> {
  const res = await fetch(`${API_BASE_URL}/api/v1/arena/battle`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      model_a: modelA,
      model_b: modelB,
      category,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Battle failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function voteInArena(
  modelA: string,
  modelB: string,
  winner: "model_a" | "model_b" | "tie",
  category = "overall",
  prompt?: string
): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/v1/arena/vote`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_a: modelA,
      model_b: modelB,
      winner,
      category,
      prompt,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Vote failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function runArenaTournament(
  models: ArenaModelTarget[],
  categories?: string[],
  promptsPerCategory = 1
): Promise<any> {
  const res = await fetch(`${API_BASE_URL}/api/v1/arena/tournament`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      models,
      categories,
      prompts_per_category: promptsPerCategory,
    }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Tournament failed (${res.status}): ${text}`);
  }
  return res.json();
}

export async function fetchModels(): Promise<ModelMetadata[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/models`);
    if (!res.ok) throw new Error("Failed to fetch models");
    const json = await res.json();
    return json.models || [];
  } catch (err) {
    console.warn("Could not fetch models, falling back to local defaults:", err);
    return [
      {
        id: "libra-llama-tied",
        name: "Libra Modern Transformer (Phase 4)",
        provider: "libra_lab",
        architecture: "Modern Transformer LM",
        context_length: 512,
        parameter_count: "467K",
        hardware_tier: "cpu-light",
        is_local: true,
        requires_gpu: false,
      },
      {
        id: "libra-mock-v1",
        name: "Libra Mock Engine v1",
        provider: "mock-provider",
        architecture: "Simulation",
        context_length: 512,
        parameter_count: "1M",
        hardware_tier: "cpu-light",
        is_local: true,
        requires_gpu: false,
      },
    ];
  }
}

export interface ConversationSummary {
  id: string;
  title: string;
  model: string;
  system_prompt?: string | null;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface StoredMessage {
  id: string;
  conversation_id: string;
  role: "system" | "user" | "assistant";
  content: string;
  token_count: number;
  created_at: string;
}

export interface ConversationDetail extends ConversationSummary {
  messages: StoredMessage[];
}

export async function fetchConversations(): Promise<ConversationSummary[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/conversations`);
    if (!res.ok) throw new Error("Failed to fetch conversations");
    return await res.json();
  } catch (err) {
    console.warn("Could not fetch conversations:", err);
    return [];
  }
}

export async function createConversation(
  title?: string,
  model?: string,
  systemPrompt?: string
): Promise<ConversationSummary | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/conversations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title,
        model: model || "libra-llama-tied",
        system_prompt: systemPrompt,
      }),
    });
    if (!res.ok) throw new Error("Failed to create conversation");
    return await res.json();
  } catch (err) {
    console.error("Error creating conversation:", err);
    return null;
  }
}

export async function fetchConversation(id: string): Promise<ConversationDetail | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/conversations/${id}`);
    if (!res.ok) throw new Error("Failed to fetch conversation detail");
    return await res.json();
  } catch (err) {
    console.error(`Error fetching conversation ${id}:`, err);
    return null;
  }
}

export async function deleteConversation(id: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/conversations/${id}`, {
      method: "DELETE",
    });
    return res.ok;
  } catch (err) {
    console.error(`Error deleting conversation ${id}:`, err);
    return false;
  }
}

export async function clearConversationMessages(id: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/conversations/${id}/messages`, {
      method: "DELETE",
    });
    return res.ok;
  } catch (err) {
    console.error(`Error clearing messages for conversation ${id}:`, err);
    return false;
  }
}

export interface RAGDocument {
  id: string;
  title: string;
  content: string;
  created_at: string;
  chunk_count: number;
  metadata: Record<string, unknown>;
}

export interface RAGChunk {
  id: string;
  doc_id: string;
  doc_title: string;
  text: string;
  char_start: number;
  char_end: number;
  token_count: number;
}

export interface RAGSearchResult {
  chunk: RAGChunk;
  score: number;
  rank: number;
  dense_score?: number | null;
  dense_rank?: number | null;
  bm25_score?: number | null;
  bm25_rank?: number | null;
  rrf_score?: number | null;
  rerank_score?: number | null;
  retrieval_mode?: string | null;
}

export interface RAGQueryOptions {
  query: string;
  top_k?: number;
  mode?: "hybrid" | "dense" | "bm25";
  alpha?: number;
  use_rrf?: boolean;
  use_reranking?: boolean;
  use_deduplication?: boolean;
  rrf_k?: number;
}

export interface RAGQueryResponse {
  query: string;
  mode?: string;
  results: RAGSearchResult[];
  augmented_prompt: string;
  total_candidates?: number;
}

export async function fetchRAGDocuments(): Promise<RAGDocument[]> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/rag/documents`);
    if (!res.ok) throw new Error("Failed to fetch RAG documents");
    return await res.json();
  } catch (err) {
    console.warn("Could not fetch RAG documents:", err);
    return [];
  }
}

export async function uploadRAGDocument(title: string, content: string): Promise<RAGDocument | null> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/rag/documents`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, content }),
    });
    if (!res.ok) throw new Error("Failed to upload document");
    return await res.json();
  } catch (err) {
    console.error("Error uploading RAG document:", err);
    return null;
  }
}

export async function deleteRAGDocument(docId: string): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/rag/documents/${docId}`, {
      method: "DELETE",
    });
    return res.ok;
  } catch (err) {
    console.error(`Error deleting RAG document ${docId}:`, err);
    return false;
  }
}

export async function queryRAG(
  options: string | RAGQueryOptions,
  legacyTopK?: number
): Promise<RAGQueryResponse | null> {
  const payload =
    typeof options === "string"
      ? { query: options, top_k: legacyTopK ?? 3, mode: "hybrid" }
      : {
          query: options.query,
          top_k: options.top_k ?? 3,
          mode: options.mode ?? "hybrid",
          alpha: options.alpha ?? 0.5,
          use_rrf: options.use_rrf ?? true,
          use_reranking: options.use_reranking ?? true,
          use_deduplication: options.use_deduplication ?? true,
          rrf_k: options.rrf_k ?? 60,
        };

  try {
    const res = await fetch(`${API_BASE_URL}/api/v1/rag/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error("Failed to query knowledge base");
    return await res.json();
  } catch (err) {
    console.error("Error querying RAG:", err);
    return null;
  }
}

export async function streamChat(
  messages: ChatMessage[],
  model: string,
  options: {
    conversationId?: string;
    useRag?: boolean;
    temperature?: number;
    maxTokens?: number;
    signal?: AbortSignal;
    onToken: (token: string) => void;
    onComplete: (telemetry: ChatTelemetry) => void;
    onError: (err: Error) => void;
  }
): Promise<void> {
  const startTime = performance.now();
  let firstTokenTime: number | null = null;
  let accumulatedTokens = 0;

  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      signal: options.signal,
      body: JSON.stringify({
        messages,
        model,
        conversation_id: options.conversationId,
        use_rag: options.useRag ?? false,
        temperature: options.temperature ?? 0.7,
        max_tokens: options.maxTokens ?? 512,
        stream: true,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Chat request failed (${response.status}): ${errText}`);
    }

    if (!response.body) {
      throw new Error("No response stream body available");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(":") || !trimmed.startsWith("data: ")) continue;

        const dataStr = trimmed.slice(6);
        if (dataStr === "[DONE]") {
          const endTime = performance.now();
          const totalLatencyMs = endTime - startTime;
          const ttftMs = firstTokenTime ? firstTokenTime - startTime : totalLatencyMs;
          const durationSec = Math.max(0.001, totalLatencyMs / 1000);
          const tps = accumulatedTokens / durationSec;

          options.onComplete({
            ttftMs,
            totalLatencyMs,
            tokensPerSecond: tps,
            totalCostUsd: 0.0,
            isFree: true,
          });
          return;
        }

        try {
          const parsed = JSON.parse(dataStr);
          if (parsed.error) {
            throw new Error(parsed.error);
          }
          const deltaContent = parsed.choices?.[0]?.delta?.content;
          if (deltaContent) {
            if (firstTokenTime === null) {
              firstTokenTime = performance.now();
            }
            accumulatedTokens++;
            options.onToken(deltaContent);
          }
        } catch {
          // ignore parsing error for individual chunk
        }
      }
    }

    const endTime = performance.now();
    const totalLatencyMs = endTime - startTime;
    const ttftMs = firstTokenTime ? firstTokenTime - startTime : totalLatencyMs;
    const durationSec = Math.max(0.001, totalLatencyMs / 1000);

    options.onComplete({
      ttftMs,
      totalLatencyMs,
      tokensPerSecond: accumulatedTokens / durationSec,
      totalCostUsd: 0.0,
      isFree: true,
    });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      const endTime = performance.now();
      const totalLatencyMs = endTime - startTime;
      const ttftMs = firstTokenTime ? firstTokenTime - startTime : totalLatencyMs;
      const durationSec = Math.max(0.001, totalLatencyMs / 1000);

      options.onComplete({
        ttftMs,
        totalLatencyMs,
        tokensPerSecond: accumulatedTokens / durationSec,
        totalCostUsd: 0.0,
        isFree: true,
      });
      return;
    }
    options.onError(err instanceof Error ? err : new Error(String(err)));
  }
}

export async function compareInArena(
  prompt: string,
  models: ArenaModelTarget[],
  temperature = 0.7,
  maxTokens = 256
): Promise<ArenaComparisonResponse> {
  const response = await fetch(`${API_BASE_URL}/api/v1/arena/compare`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      prompt,
      models,
      temperature,
      max_tokens: maxTokens,
    }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Arena comparison failed (${response.status}): ${errText}`);
  }

  return response.json();
}

export async function streamTelemetry(
  prompt: string,
  options: {
    maxTokens?: number;
    temperature?: number;
    topK?: number;
    candidateTopK?: number;
    signal?: AbortSignal;
    onToken?: (token: TokenTelemetryData) => void;
    onComplete?: (summary: SequenceTelemetryData) => void;
    onError?: (err: Error) => void;
  }
): Promise<void> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/telemetry/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: options.signal,
      body: JSON.stringify({
        prompt,
        max_tokens: options.maxTokens ?? 24,
        temperature: options.temperature ?? 0.7,
        top_k: options.topK ?? 40,
        candidate_top_k: options.candidateTopK ?? 5,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Telemetry request failed (${response.status}): ${errText}`);
    }

    if (!response.body) {
      throw new Error("No response body available for telemetry stream");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let currentEvent = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(":")) continue;

        if (trimmed.startsWith("event: ")) {
          currentEvent = trimmed.slice(7).trim();
          continue;
        }

        if (trimmed.startsWith("data: ")) {
          const dataStr = trimmed.slice(6).trim();
          if (dataStr === "[DONE]") {
            return;
          }

          try {
            const parsed = JSON.parse(dataStr);
            if (currentEvent === "token" && options.onToken) {
              options.onToken(parsed as TokenTelemetryData);
            } else if (currentEvent === "done" && options.onComplete) {
              options.onComplete(parsed as SequenceTelemetryData);
            }
          } catch (jsonErr) {
            console.warn("Failed to parse telemetry stream data chunk:", jsonErr);
          }
        }
      }
    }
  } catch (err) {
    if (options.onError) {
      options.onError(err instanceof Error ? err : new Error(String(err)));
    }
  }
}

