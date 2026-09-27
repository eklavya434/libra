/**
 * Libra Module Status Configuration
 *
 * Single source of truth for maturity and implementation status across
 * the /chat sidebar and the landing page feature cards.
 *
 * Core project rule: "No fabricated metrics" / Honesty First.
 * Clearly label what is shipped, what is in active development,
 * and what is an experimental research module.
 */

export type ModuleMaturity = 'stable' | 'in-progress' | 'experimental';

export interface ModuleStatus {
  id: string;
  name: string;
  maturity: ModuleMaturity;
  label: string; // e.g., "Shipped", "In active development", "Experimental"
  shortBadge: string; // e.g., "Live", "Dev", "Exp"
  tier: 'core' | 'advanced';
  description: string;
}

export const MODULE_REGISTRY: Record<string, ModuleStatus> = {
  // Core Tier (4 primary visitor entry points)
  chat: {
    id: 'chat',
    name: 'Interactive Chat',
    maturity: 'stable',
    label: 'Shipped',
    shortBadge: 'Live',
    tier: 'core',
    description: 'Multi-turn conversational assistant with SSE token streaming and guest sessions.',
  },
  arena: {
    id: 'arena',
    name: 'Model Arena',
    maturity: 'stable',
    label: 'Shipped',
    shortBadge: 'Live',
    tier: 'core',
    description: 'Side-by-side generation comparison profiling TTFT, token velocity, and output quality.',
  },
  rag: {
    id: 'rag',
    name: 'Knowledge Base (RAG)',
    maturity: 'stable',
    label: 'Shipped',
    shortBadge: 'Live',
    tier: 'core',
    description: 'Dense embeddings + Okapi BM25 hybrid search with Reciprocal Rank Fusion.',
  },
  notebook: {
    id: 'notebook',
    name: 'Code Notebook',
    maturity: 'stable',
    label: 'Shipped',
    shortBadge: 'Live',
    tier: 'core',
    description: 'Stateful Python execution sandbox with persistent cell state and SVG plotting.',
  },

  // Advanced Labs Tier (14 research and evolution modules)
  corpus: {
    id: 'corpus',
    name: 'Corpus & SFT Lab',
    maturity: 'in-progress',
    label: 'In active development',
    shortBadge: 'Dev',
    tier: 'advanced',
    description: 'Instruction fine-tuning dataset curation, ChatML packing, and prompt loss masking.',
  },
  security: {
    id: 'security',
    name: 'Security Lab',
    maturity: 'stable',
    label: 'Shipped',
    shortBadge: 'Live',
    tier: 'advanced',
    description: 'PromptGuard injection defense, SecretScanner DLP, and token-bucket rate limiting.',
  },
  observability: {
    id: 'observability',
    name: 'Observability & Traces',
    maturity: 'stable',
    label: 'Shipped',
    shortBadge: 'Live',
    tier: 'advanced',
    description: 'OpenTelemetry distributed spans, token latency telemetry, and surprisal distributions.',
  },
  batch: {
    id: 'batch',
    name: 'Batch Inference',
    maturity: 'in-progress',
    label: 'In active development',
    shortBadge: 'Dev',
    tier: 'advanced',
    description: 'Dynamic sequence binning and asynchronous worker queues for high-throughput jobs.',
  },
  document: {
    id: 'document',
    name: 'Document OCR',
    maturity: 'in-progress',
    label: 'In active development',
    shortBadge: 'Dev',
    tier: 'advanced',
    description: 'Layout-aware document chunking, visual question answering, and formula parsing.',
  },
  long_context: {
    id: 'long_context',
    name: 'Long Context & NIAH',
    maturity: 'in-progress',
    label: 'In active development',
    shortBadge: 'Dev',
    tier: 'advanced',
    description: 'Needle-in-a-haystack synthetic stress testing and attention sink compaction.',
  },
  mcts: {
    id: 'mcts',
    name: 'MCTS Reasoning',
    maturity: 'experimental',
    label: 'Experimental',
    shortBadge: 'Exp',
    tier: 'advanced',
    description: 'Monte Carlo Tree Search with Process Reward Model (PRM) branch selection.',
  },
  distillation: {
    id: 'distillation',
    name: 'Distillation Lab',
    maturity: 'experimental',
    label: 'Experimental',
    shortBadge: 'Exp',
    tier: 'advanced',
    description: 'Teacher-student soft logit KL-divergence transfer and progressive layer pruning.',
  },
  moe: {
    id: 'moe',
    name: 'MoE Sparse Lab',
    maturity: 'experimental',
    label: 'Experimental',
    shortBadge: 'Exp',
    tier: 'advanced',
    description: 'Top-k noisy gating sparse Mixture of Experts with load-balancing auxiliary loss.',
  },
  medusa: {
    id: 'medusa',
    name: 'Medusa Speculative',
    maturity: 'experimental',
    label: 'Experimental',
    shortBadge: 'Exp',
    tier: 'advanced',
    description: 'Multi-head residual draft heads with parallel prefix-tree candidate verification.',
  },
  kto: {
    id: 'kto',
    name: 'KTO Alignment',
    maturity: 'in-progress',
    label: 'In active development',
    shortBadge: 'Dev',
    tier: 'advanced',
    description: 'Kahneman-Tversky prospect-theoretic alignment from binary desirability signal.',
  },
  verifiable_search: {
    id: 'verifiable_search',
    name: 'Verifiable Search',
    maturity: 'experimental',
    label: 'Experimental',
    shortBadge: 'Exp',
    tier: 'advanced',
    description: 'Step-level reasoning verification with backtracking on flawed intermediate deductions.',
  },
  self_rewarding: {
    id: 'self_rewarding',
    name: 'Self-Rewarding',
    maturity: 'experimental',
    label: 'Experimental',
    shortBadge: 'Exp',
    tier: 'advanced',
    description: 'Iterative self-alignment flywheel with LLM-as-a-Judge debiasing filters.',
  },
  grand_capstone: {
    id: 'grand_capstone',
    name: 'Grand Capstone',
    maturity: 'in-progress',
    label: 'In active development',
    shortBadge: 'Dev',
    tier: 'advanced',
    description: 'Master autonomous self-evolution orchestrator and architectural verification suite.',
  },
};

/**
 * Landing page feature mapping
 */
export const LANDING_FEATURES_STATUS: Record<string, ModuleMaturity> = {
  'Chat Assistant': 'stable',
  'Transformer Lab': 'stable',
  'Arena & Evaluation': 'stable',
  'RAG Knowledge Base': 'stable',
  'Stateful Data Notebook': 'stable',
  'Security & Honesty First': 'stable',
};

/**
 * Styling helpers for consistent maturity badges across the entire site
 */
export function getMaturityStyle(maturity: ModuleMaturity) {
  switch (maturity) {
    case 'stable':
      return {
        badgeText: 'Shipped',
        dotColor: 'bg-emerald-400',
        textColor: 'text-emerald-300',
        bgColor: 'bg-emerald-500/10',
        borderColor: 'border-emerald-500/20',
      };
    case 'in-progress':
      return {
        badgeText: 'In active development',
        dotColor: 'bg-amber-400',
        textColor: 'text-amber-300',
        bgColor: 'bg-amber-500/10',
        borderColor: 'border-amber-500/20',
      };
    case 'experimental':
      return {
        badgeText: 'Experimental',
        dotColor: 'bg-purple-400',
        textColor: 'text-purple-300',
        bgColor: 'bg-purple-500/10',
        borderColor: 'border-purple-500/20',
      };
  }
}
