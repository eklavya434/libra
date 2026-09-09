'use client';

import React, { useState, useEffect } from 'react';
import {
  Network,
  Cpu,
  Layers,
  Sparkles,
  Zap,
  BarChart2,
  Sliders,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  TrendingDown,
  Activity,
} from 'lucide-react';

interface Preset {
  id: string;
  name: string;
  description: string;
  num_experts: number;
  top_k: number;
  aux_loss_coef: number;
  noisy_gating: boolean;
}

interface TokenTrace {
  position: number;
  token_id: number;
  token_str: string;
  layers: Array<{
    layer_idx: number;
    selected_experts: Array<{
      expert_id: number;
      weight: number;
    }>;
  }>;
}

interface LayerStat {
  layer_idx: number;
  expert_counts: number[];
  expert_fractions: number[];
  entropy: number;
  max_possible_entropy: number;
  uniformity_score_pct: number;
}

interface ParameterEfficiency {
  total_parameters: number;
  active_parameters_per_token: number;
  sparsity_ratio: number;
  compute_savings_pct: number;
  num_experts: number;
  num_experts_per_tok: number;
  n_layers: number;
  d_model: number;
}

const EXPERT_COLORS = [
  { bg: 'bg-emerald-500/20', text: 'text-emerald-400', border: 'border-emerald-500/40', bar: 'bg-emerald-500' },
  { bg: 'bg-amber-500/20', text: 'text-amber-400', border: 'border-amber-500/40', bar: 'bg-amber-500' },
  { bg: 'bg-cyan-500/20', text: 'text-cyan-400', border: 'border-cyan-500/40', bar: 'bg-cyan-500' },
  { bg: 'bg-purple-500/20', text: 'text-purple-400', border: 'border-purple-500/40', bar: 'bg-purple-500' },
];

export default function MoEView() {
  const [activeSubTab, setActiveSubTab] = useState<'routing' | 'utilization' | 'trainer'>('routing');
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('mixtral_top2');

  // Routing State
  const [prompt, setPrompt] = useState<string>(
    'Mixture of experts dynamically routes tokens to specialized feed-forward sub-networks.'
  );
  const [isLoadingRouting, setIsLoadingRouting] = useState<boolean>(false);
  const [selectedTokenIdx, setSelectedTokenIdx] = useState<number>(0);
  const [routingData, setRoutingData] = useState<{
    prompt_tokens_count: number;
    num_experts: number;
    top_k: number;
    aux_loss: number;
    tokens: TokenTrace[];
    layer_stats: LayerStat[];
    parameter_efficiency: ParameterEfficiency;
  } | null>(null);

  // Training State
  const [trainSteps, setTrainSteps] = useState<number>(10);
  const [auxCoef, setAuxCoef] = useState<number>(0.02);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainData, setTrainData] = useState<{
    history: Array<{
      step: number;
      total_loss: number;
      task_loss: number;
      aux_loss: number;
      cv_imbalance: number;
      expert_fractions: number[];
    }>;
    initial_total_loss: number;
    final_total_loss: number;
    initial_cv: number;
    final_cv: number;
    balance_improved: boolean;
  } | null>(null);

  // Utilization State
  const [isCheckingUtil, setIsCheckingUtil] = useState<boolean>(false);
  const [utilizationData, setUtilizationData] = useState<{
    total_tokens_evaluated: number;
    total_expert_dispatches: number;
    expert_counts: number[];
    expert_fractions: number[];
    coefficient_of_variation: number;
    is_balanced: boolean;
    starved_experts: number[];
    parameter_efficiency: ParameterEfficiency;
  } | null>(null);

  // Load Presets
  useEffect(() => {
    fetch('/api/v1/moe/presets')
      .then((res) => res.json())
      .then((data) => {
        if (data.presets) setPresets(data.presets);
      })
      .catch(() => {});
  }, []);

  // Initial routing load
  useEffect(() => {
    handleRunRouting();
  }, []);

  const handleRunRouting = async () => {
    setIsLoadingRouting(true);
    try {
      const res = await fetch('/api/v1/moe/forward', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt }),
      });
      const data = await res.json();
      setRoutingData(data);
      if (data.tokens && data.tokens.length > 0) {
        setSelectedTokenIdx(0);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoadingRouting(false);
    }
  };

  const handleTrainMoE = async () => {
    setIsTraining(true);
    try {
      const res = await fetch('/api/v1/moe/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          steps: trainSteps,
          aux_loss_coef: auxCoef,
          lr: 0.005,
        }),
      });
      const data = await res.json();
      setTrainData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsTraining(false);
    }
  };

  const handleFetchUtilization = async () => {
    setIsCheckingUtil(true);
    try {
      const res = await fetch('/api/v1/moe/utilization');
      const data = await res.json();
      setUtilizationData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsCheckingUtil(false);
    }
  };

  const selectedToken =
    routingData && routingData.tokens && routingData.tokens[selectedTokenIdx]
      ? routingData.tokens[selectedTokenIdx]
      : null;

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 text-slate-100 overflow-y-auto">
      {/* Header */}
      <div className="border-b border-slate-800/80 bg-slate-900/60 px-6 py-4 backdrop-blur-sm sticky top-0 z-10">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-purple-600 to-indigo-500 flex items-center justify-center text-white shadow-lg shadow-purple-500/20">
              <Network className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-semibold text-slate-100 tracking-tight">
                  Sparse Mixture of Experts (MoE) Lab
                </h1>
                <span className="text-[10px] font-mono uppercase bg-purple-500/20 text-purple-300 border border-purple-500/30 px-2 py-0.5 rounded-full">
                  Phase 45
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Top-K sparse gating, conditional feed-forward dispatch, and Switch/Shazeer load-balancing auxiliary loss
              </p>
            </div>
          </div>

          {/* Sub-tab Switcher */}
          <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-1 text-xs">
            <button
              onClick={() => setActiveSubTab('routing')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeSubTab === 'routing'
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Network className="w-3.5 h-3.5" />
              <span>Token Routing</span>
            </button>
            <button
              onClick={() => {
                setActiveSubTab('utilization');
                if (!utilizationData) handleFetchUtilization();
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeSubTab === 'utilization'
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <BarChart2 className="w-3.5 h-3.5" />
              <span>Expert Load & Balance</span>
            </button>
            <button
              onClick={() => setActiveSubTab('trainer')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeSubTab === 'trainer'
                  ? 'bg-purple-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              <span>Auxiliary Loss Trainer</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Container */}
      <div className="p-6 space-y-6 max-w-7xl mx-auto w-full">
        {/* Parameter Decoupling Bar */}
        {routingData?.parameter_efficiency && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400">Total Parameter Capacity</span>
              <div className="text-lg font-mono font-bold text-slate-200 mt-1">
                {routingData.parameter_efficiency.total_parameters.toLocaleString()}
              </div>
              <span className="text-[10px] text-purple-400 font-mono">
                {routingData.num_experts} Experts (SwiGLU)
              </span>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400">Active Parameters per Token</span>
              <div className="text-lg font-mono font-bold text-emerald-400 mt-1">
                {routingData.parameter_efficiency.active_parameters_per_token.toLocaleString()}
              </div>
              <span className="text-[10px] text-emerald-400 font-mono">
                Top-{routingData.top_k} Selected ({routingData.parameter_efficiency.compute_savings_pct}% Compute Saved)
              </span>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400">Parameter Decoupling Ratio</span>
              <div className="text-lg font-mono font-bold text-cyan-400 mt-1">
                {routingData.parameter_efficiency.sparsity_ratio}x
              </div>
              <span className="text-[10px] text-slate-400 font-mono">
                Capacity / Compute FLOPs
              </span>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400">Load Balance Aux Loss</span>
              <div className="text-lg font-mono font-bold text-amber-400 mt-1">
                {routingData.aux_loss.toFixed(4)}
              </div>
              <span className="text-[10px] text-slate-400 font-mono">
                L_aux = α · E · sum(f_i · P_i)
              </span>
            </div>
          </div>
        )}

        {/* SUB-TAB 1: TOKEN ROUTING VISUALIZER */}
        {activeSubTab === 'routing' && (
          <div className="space-y-6">
            {/* Prompt Input & Presets */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-3">
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Network className="w-4 h-4 text-purple-400" />
                  Prompt Token Dispatch Sandbox
                </h2>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-slate-400 mr-1">Presets:</span>
                  {presets.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setSelectedPresetId(p.id);
                        if (p.id === 'switch_top1') {
                          setPrompt('Switch transformer activates exactly one expert per token.');
                        } else {
                          setPrompt('Mixture of experts dynamically routes tokens to specialized feed-forward sub-networks.');
                        }
                      }}
                      className={`px-2.5 py-1 rounded text-xs transition-all ${
                        selectedPresetId === p.id
                          ? 'bg-purple-600/30 text-purple-200 border border-purple-500/50'
                          : 'bg-slate-800/60 hover:bg-slate-800 text-slate-300 border border-slate-700/50'
                      }`}
                    >
                      {p.name.split(' ')[0]}
                    </button>
                  ))}
                </div>
              </div>

              <div className="flex gap-2">
                <input
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-purple-500 font-mono"
                  placeholder="Type an input prompt..."
                />
                <button
                  onClick={handleRunRouting}
                  disabled={isLoadingRouting}
                  className="bg-purple-600 hover:bg-purple-500 disabled:bg-purple-900/50 text-white px-5 py-2 rounded-lg text-xs font-medium flex items-center gap-1.5 shadow-md shadow-purple-600/20 cursor-pointer"
                >
                  {isLoadingRouting ? (
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Play className="w-3.5 h-3.5 fill-current" />
                  )}
                  <span>Route Tokens</span>
                </button>
              </div>
            </div>

            {/* Token Tape with Top-K Badges */}
            {routingData && routingData.tokens && (
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-purple-400" />
                    Interactive Token Stream (Click token to inspect expert weights)
                  </span>
                  <span className="text-[11px] text-slate-400 font-mono">
                    {routingData.prompt_tokens_count} Tokens Dispatched
                  </span>
                </div>

                <div className="flex flex-wrap gap-2 p-3 bg-slate-950/80 rounded-xl border border-slate-850">
                  {routingData.tokens.map((tok, idx) => {
                    const isSelected = selectedTokenIdx === idx;
                    const primaryExpert =
                      tok.layers[0]?.selected_experts[0]?.expert_id ?? 0;
                    const colorScheme = EXPERT_COLORS[primaryExpert % EXPERT_COLORS.length];

                    return (
                      <button
                        key={idx}
                        onClick={() => setSelectedTokenIdx(idx)}
                        className={`flex flex-col items-center px-2.5 py-1.5 rounded-lg text-xs font-mono transition-all border ${
                          isSelected
                            ? 'bg-purple-600/30 border-purple-400 text-white ring-2 ring-purple-500/40'
                            : `${colorScheme.bg} ${colorScheme.border} ${colorScheme.text} hover:scale-105`
                        }`}
                      >
                        <span className="font-sans font-medium text-slate-100">
                          {tok.token_str}
                        </span>
                        <div className="flex gap-1 mt-1">
                          {tok.layers[0]?.selected_experts.map((exp, eIdx) => (
                            <span
                              key={eIdx}
                              className={`text-[9px] px-1 py-0.2 rounded font-mono ${
                                EXPERT_COLORS[exp.expert_id % EXPERT_COLORS.length].bg
                              } ${EXPERT_COLORS[exp.expert_id % EXPERT_COLORS.length].text}`}
                            >
                              E{exp.expert_id}
                            </span>
                          ))}
                        </div>
                      </button>
                    );
                  })}
                </div>

                {/* Selected Token Detail Inspector */}
                {selectedToken && (
                  <div className="bg-slate-950 border border-slate-800 rounded-xl p-4 space-y-3">
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <div className="flex items-center gap-2 text-xs">
                        <span className="text-slate-400">Inspecting Token:</span>
                        <span className="font-bold text-purple-300 font-mono text-sm">
                          &quot;{selectedToken.token_str}&quot;
                        </span>
                        <span className="text-[10px] text-slate-400 font-mono">
                          (Position #{selectedToken.position}, Token ID: {selectedToken.token_id})
                        </span>
                      </div>
                      <span className="text-[10px] bg-purple-500/20 text-purple-300 border border-purple-500/30 px-2 py-0.5 rounded">
                        Top-{routingData.top_k} Gating
                      </span>
                    </div>

                    {/* Per Layer Gating Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {selectedToken.layers.map((layer) => (
                        <div
                          key={layer.layer_idx}
                          className="bg-slate-900/60 border border-slate-800/80 rounded-lg p-3 space-y-2"
                        >
                          <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                            <Layers className="w-3.5 h-3.5 text-indigo-400" />
                            Transformer Block #{layer.layer_idx} MoE Router
                          </span>

                          <div className="space-y-1.5">
                            {layer.selected_experts.map((exp) => {
                              const color = EXPERT_COLORS[exp.expert_id % EXPERT_COLORS.length];
                              return (
                                <div key={exp.expert_id} className="space-y-0.5">
                                  <div className="flex justify-between text-xs font-mono">
                                    <span className={color.text}>Expert E{exp.expert_id}</span>
                                    <span className="text-slate-200">
                                      {(exp.weight * 100).toFixed(1)}% Gating Weight
                                    </span>
                                  </div>
                                  <div className="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                    <div
                                      className={`${color.bar} h-full rounded-full transition-all duration-300`}
                                      style={{ width: `${Math.min(100, exp.weight * 100)}%` }}
                                    ></div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* SUB-TAB 2: EXPERT LOAD & UTILIZATION */}
        {activeSubTab === 'utilization' && (
          <div className="space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-cyan-400" />
                  Expert Load Distribution & Starvation Analysis
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Evaluates whether tokens are uniformly distributed across all experts or if specific experts are starving.
                </p>
              </div>
              <button
                onClick={handleFetchUtilization}
                disabled={isCheckingUtil}
                className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-cyan-900/50 text-white px-4 py-2 rounded-lg text-xs font-medium flex items-center gap-1.5 shadow-md shadow-cyan-600/20 cursor-pointer"
              >
                {isCheckingUtil ? (
                  <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Play className="w-3.5 h-3.5 fill-current" />
                )}
                <span>Run Validation Suite</span>
              </button>
            </div>

            {utilizationData && (
              <div className="space-y-6">
                {/* Balance Status Banner */}
                <div
                  className={`p-4 rounded-xl border flex items-center justify-between ${
                    utilizationData.is_balanced
                      ? 'bg-emerald-950/20 border-emerald-500/30 text-emerald-300'
                      : 'bg-amber-950/20 border-amber-500/30 text-amber-300'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {utilizationData.is_balanced ? (
                      <CheckCircle2 className="w-5 h-5 text-emerald-400" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-amber-400" />
                    )}
                    <div>
                      <h3 className="text-xs font-semibold">
                        {utilizationData.is_balanced
                          ? 'Optimal Expert Balance Detected'
                          : 'Potential Expert Imbalance'}
                      </h3>
                      <p className="text-[11px] opacity-80 mt-0.5">
                        Coefficient of Variation (CV): {utilizationData.coefficient_of_variation} (Threshold: &lt; 0.35)
                      </p>
                    </div>
                  </div>
                  <div className="text-xs font-mono">
                    {utilizationData.starved_experts.length === 0 ? (
                      <span className="text-emerald-400 font-medium">0 Starved Experts</span>
                    ) : (
                      <span className="text-amber-400 font-medium">
                        Starved: {utilizationData.starved_experts.map((e) => `E${e}`).join(', ')}
                      </span>
                    )}
                  </div>
                </div>

                {/* 4 Expert Cards */}
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {utilizationData.expert_counts.map((count, expId) => {
                    const frac = utilizationData.expert_fractions[expId];
                    const color = EXPERT_COLORS[expId % EXPERT_COLORS.length];
                    const isStarved = utilizationData.starved_experts.includes(expId);

                    return (
                      <div
                        key={expId}
                        className={`bg-slate-900/70 border rounded-xl p-4 flex flex-col justify-between ${
                          isStarved ? 'border-rose-500/40' : 'border-slate-800'
                        }`}
                      >
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <span className={`text-xs font-semibold uppercase font-mono ${color.text}`}>
                              Expert E{expId}
                            </span>
                            <span
                              className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                                isStarved
                                  ? 'bg-rose-500/20 text-rose-300'
                                  : 'bg-slate-800 text-slate-300'
                              }`}
                            >
                              {isStarved ? 'Starved (&lt;5%)' : 'Healthy'}
                            </span>
                          </div>
                          <div className="text-xl font-mono font-bold text-slate-100">
                            {(frac * 100).toFixed(1)}%
                          </div>
                          <span className="text-[11px] text-slate-400 font-mono">
                            {count} Total Tokens Routed
                          </span>
                        </div>

                        <div className="mt-3 pt-3 border-t border-slate-800/80 space-y-1">
                          <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                            <div
                              className={`${color.bar} h-full rounded-full transition-all duration-500`}
                              style={{ width: `${Math.min(100, frac * 100)}%` }}
                            ></div>
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* SUB-TAB 3: AUXILIARY LOSS TRAINER */}
        {activeSubTab === 'trainer' && (
          <div className="space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="border-b border-slate-800 pb-3">
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-purple-400" />
                  Load-Balancing Auxiliary Loss Optimizer
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Trains the MoE model with Switch/Shazeer auxiliary loss: L_aux = α · E · sum(f_i · P_i).
                  Minimizing L_aux eliminates routing collapse and guides the router to utilize all experts.
                </p>
              </div>

              {/* Sliders */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-850">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-400 font-medium">Auxiliary Loss Weight (α)</span>
                    <span className="font-mono text-purple-400 font-semibold">{auxCoef.toFixed(3)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="0.1"
                    step="0.005"
                    value={auxCoef}
                    onChange={(e) => setAuxCoef(parseFloat(e.target.value))}
                    className="w-full accent-purple-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-400 mt-1">
                    {auxCoef === 0.0 ? 'Ablation: No balancing penalty (expert collapse risk)' : 'Active load balancing penalty'}
                  </p>
                </div>

                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-850">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-400 font-medium">Training Steps</span>
                    <span className="font-mono text-purple-400 font-semibold">{trainSteps}</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="25"
                    step="5"
                    value={trainSteps}
                    onChange={(e) => setTrainSteps(parseInt(e.target.value))}
                    className="w-full accent-purple-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-400 mt-1">CPU training step iterations</p>
                </div>
              </div>

              <div className="flex justify-end pt-2">
                <button
                  onClick={handleTrainMoE}
                  disabled={isTraining}
                  className="bg-purple-600 hover:bg-purple-500 disabled:bg-purple-900/50 text-white px-5 py-2 rounded-lg text-xs font-medium flex items-center gap-2 shadow-md shadow-purple-600/20 cursor-pointer"
                >
                  {isTraining ? (
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Play className="w-3.5 h-3.5 fill-current" />
                  )}
                  <span>Execute Training Loop</span>
                </button>
              </div>
            </div>

            {/* Training Results */}
            {trainData && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                    <span className="text-xs text-slate-400">Total Loss Convergence</span>
                    <div className="flex items-baseline gap-2 mt-1">
                      <span className="text-lg font-mono font-bold text-slate-200">
                        {trainData.initial_total_loss.toFixed(3)}
                      </span>
                      <ArrowRight className="w-3 h-3 text-slate-400" />
                      <span className="text-lg font-mono font-bold text-emerald-400">
                        {trainData.final_total_loss.toFixed(3)}
                      </span>
                    </div>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                    <span className="text-xs text-slate-400">Routing Imbalance (CV)</span>
                    <div className="flex items-baseline gap-2 mt-1">
                      <span className="text-lg font-mono font-bold text-slate-200">
                        {trainData.initial_cv}
                      </span>
                      <ArrowRight className="w-3 h-3 text-slate-400" />
                      <span className="text-lg font-mono font-bold text-cyan-400">
                        {trainData.final_cv}
                      </span>
                    </div>
                    <span className="text-[10px] text-cyan-400 font-mono">
                      {trainData.balance_improved ? 'Load balance improved' : 'Balance maintained'}
                    </span>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                    <span className="text-xs text-slate-400">Training Status</span>
                    <div className="text-lg font-mono font-bold text-purple-400 mt-1">
                      {trainData.history.length} Steps Complete
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">
                      AdamW (100% CPU Offline)
                    </span>
                  </div>
                </div>

                {/* Telemetry Table */}
                <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 space-y-3">
                  <h3 className="text-xs font-semibold text-slate-200">
                    Step-by-Step Loss & Routing Telemetry
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="text-[11px] uppercase bg-slate-950/80 text-slate-400 font-mono border-b border-slate-800">
                        <tr>
                          <th className="py-2 px-3">Step</th>
                          <th className="py-2 px-3">Total Loss</th>
                          <th className="py-2 px-3">Task CE Loss</th>
                          <th className="py-2 px-3">Aux Loss</th>
                          <th className="py-2 px-3">Imbalance CV</th>
                          <th className="py-2 px-3">Expert Fractions [E0, E1, E2, E3]</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-850 font-mono text-xs">
                        {trainData.history.map((row) => (
                          <tr key={row.step} className="hover:bg-slate-800/30">
                            <td className="py-2 px-3 font-semibold text-slate-400">#{row.step}</td>
                            <td className="py-2 px-3 text-purple-300 font-bold">{row.total_loss}</td>
                            <td className="py-2 px-3 text-slate-200">{row.task_loss}</td>
                            <td className="py-2 px-3 text-amber-300">{row.aux_loss}</td>
                            <td className="py-2 px-3 text-cyan-300">{row.cv_imbalance}</td>
                            <td className="py-2 px-3 text-slate-400">
                              [{row.expert_fractions.join(', ')}]
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
