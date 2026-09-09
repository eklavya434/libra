'use client';

import React, { useState, useEffect } from 'react';
import {
  CheckCheck,
  Play,
  Sliders,
  RotateCcw,
  Sparkles,
  BarChart2,
  TrendingUp,
  Cpu,
  Layers,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  Activity,
  GitBranch,
  Search,
  Zap,
} from 'lucide-react';

interface Preset {
  id: string;
  name: string;
  prompt: string;
  expected_answer: string;
  description: string;
  beam_width: number;
  step_prune_threshold: number;
}

interface StepNode {
  id: string;
  parent_id: string | null;
  content: string;
  depth: number;
  step_score: number;
  cumulative_score: number;
  status: 'active' | 'accepted' | 'pruned' | 'terminal';
  rationale: string;
  errors: string[];
  num_children: number;
}

interface SolveResponse {
  prompt: string;
  optimal_trajectory: StepNode[];
  final_answer: string;
  is_verified: boolean;
  total_steps_explored: number;
  pruned_branches_count: number;
  backtracks_count: number;
  compute_savings_pct: number;
  tree_nodes: StepNode[];
  tree_edges: { from: string; to: string }[];
}

interface MethodMetrics {
  method: string;
  accuracy: number;
  avg_steps_generated: number;
  early_prune_rate: number;
  compute_savings_pct: number;
  avg_latency_ms: number;
  notes: string;
}

interface BenchmarkResponse {
  methods: MethodMetrics[];
  winner: string;
  verifiable_compute_savings: string;
}

export default function VerifiableSearchView() {
  const [activeSubTab, setActiveSubTab] = useState<'solver' | 'benchmark' | 'theory'>('solver');
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('compound_arithmetic');

  // Solver Configuration
  const [prompt, setPrompt] = useState<string>('Calculate (15 * 4) + (24 / 3) - 17');
  const [beamWidth, setBeamWidth] = useState<number>(3);
  const [pruneThreshold, setPruneThreshold] = useState<number>(0.55);
  const [maxDepth, setMaxDepth] = useState<number>(5);

  const [loading, setLoading] = useState(false);
  const [solveResult, setSolveResult] = useState<SolveResponse | null>(null);
  const [selectedNode, setSelectedNode] = useState<StepNode | null>(null);

  // Benchmark State
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResponse | null>(null);

  useEffect(() => {
    fetch('/api/v1/verifiable-search/presets')
      .then((res) => res.json())
      .then((data) => {
        if (data.presets) {
          setPresets(data.presets);
        }
      })
      .catch((err) => console.error('Failed to load presets', err));
  }, []);

  const handleSelectPreset = (pId: string) => {
    setSelectedPresetId(pId);
    const p = presets.find((item) => item.id === pId);
    if (!p) return;
    setPrompt(p.prompt);
    setBeamWidth(p.beam_width);
    setPruneThreshold(p.step_prune_threshold);
  };

  const handleRunSolve = async () => {
    setLoading(true);
    setSelectedNode(null);
    try {
      const res = await fetch('/api/v1/verifiable-search/solve', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          beam_width: beamWidth,
          max_depth: maxDepth,
          step_prune_threshold: pruneThreshold,
          branching_factor: 2,
        }),
      });
      if (res.ok) {
        const data: SolveResponse = await res.json();
        setSolveResult(data);
        if (data.optimal_trajectory.length > 0) {
          setSelectedNode(data.optimal_trajectory[data.optimal_trajectory.length - 1]);
        }
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    setBenchmarkLoading(true);
    try {
      const res = await fetch('/api/v1/verifiable-search/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          n_best_of_n: 3,
        }),
      });
      if (res.ok) {
        const data: BenchmarkResponse = await res.json();
        setBenchmarkResult(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setBenchmarkLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 text-slate-100 overflow-y-auto">
      {/* Top Header */}
      <div className="border-b border-slate-800 bg-slate-900/60 p-6 backdrop-blur">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-emerald-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
              <CheckCheck className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-white">Verifiable Search & PRM Lab</h1>
                <span className="text-xs bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 px-2 py-0.5 rounded-full font-mono">
                  Phase 48
                </span>
                <span className="text-xs bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded-full font-mono">
                  SV-Search
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Process Reward Model (PRM) Guided Beam Search with Early Pruning & Frontier Backtracking
              </p>
            </div>
          </div>

          {/* Sub-tab Switcher */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
            <button
              onClick={() => setActiveSubTab('solver')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                activeSubTab === 'solver'
                  ? 'bg-cyan-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <GitBranch className="w-3.5 h-3.5" />
              <span>Verifiable Tree Search</span>
            </button>
            <button
              onClick={() => setActiveSubTab('benchmark')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                activeSubTab === 'benchmark'
                  ? 'bg-emerald-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <BarChart2 className="w-3.5 h-3.5" />
              <span>Search Arena</span>
            </button>
            <button
              onClick={() => setActiveSubTab('theory')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                activeSubTab === 'theory'
                  ? 'bg-purple-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              <span>PRM Architecture</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="p-6 space-y-6 max-w-7xl w-full mx-auto">
        {/* Preset Selector */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-cyan-400" />
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Educational Presets</span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {presets.map((preset) => (
              <button
                key={preset.id}
                onClick={() => handleSelectPreset(preset.id)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
                  selectedPresetId === preset.id
                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/50 font-medium'
                    : 'bg-slate-800/60 text-slate-400 border-slate-700 hover:text-slate-200'
                }`}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>

        {/* Sub-Tab 1: Verifiable Tree Search */}
        {activeSubTab === 'solver' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column: Search Config & Problem Input */}
            <div className="space-y-6 lg:col-span-1">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center gap-2 text-sm font-semibold text-white">
                  <Search className="w-4 h-4 text-cyan-400" />
                  <span>Search Parameters</span>
                </div>

                <div className="space-y-3 text-xs">
                  <div>
                    <label className="block text-slate-400 mb-1">Reasoning Prompt</label>
                    <textarea
                      rows={3}
                      value={prompt}
                      onChange={(e) => setPrompt(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono focus:border-cyan-500 focus:outline-none"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Beam Width (K)</span>
                      <span className="font-mono text-cyan-300">{beamWidth}</span>
                    </div>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      step="1"
                      value={beamWidth}
                      onChange={(e) => setBeamWidth(parseInt(e.target.value))}
                      className="w-full accent-cyan-500"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>PRM Prune Threshold (tau)</span>
                      <span className="font-mono text-emerald-300">{pruneThreshold.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.30"
                      max="0.85"
                      step="0.05"
                      value={pruneThreshold}
                      onChange={(e) => setPruneThreshold(parseFloat(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Max Reasoning Depth</span>
                      <span className="font-mono text-slate-300">{maxDepth}</span>
                    </div>
                    <input
                      type="range"
                      min="2"
                      max="7"
                      step="1"
                      value={maxDepth}
                      onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                      className="w-full accent-slate-500"
                    />
                  </div>
                </div>

                <button
                  onClick={handleRunSolve}
                  disabled={loading}
                  className="w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white rounded-lg font-medium text-xs flex items-center justify-center gap-2 transition-all shadow-md shadow-cyan-600/20"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>{loading ? 'Verifying & Searching...' : 'Execute Verifiable Search'}</span>
                </button>
              </div>

              {/* Telemetry Summary Card */}
              {solveResult && (
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-white">
                    <Activity className="w-4 h-4 text-emerald-400" />
                    <span>Search Telemetry</span>
                  </div>

                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <div className="text-[10px] text-slate-400">Steps Explored</div>
                      <div className="font-mono font-bold text-white text-sm mt-0.5">{solveResult.total_steps_explored}</div>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <div className="text-[10px] text-slate-400">Pruned Branches</div>
                      <div className="font-mono font-bold text-rose-400 text-sm mt-0.5">{solveResult.pruned_branches_count}</div>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <div className="text-[10px] text-slate-400">Backtracks Triggered</div>
                      <div className="font-mono font-bold text-amber-400 text-sm mt-0.5">{solveResult.backtracks_count}</div>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <div className="text-[10px] text-slate-400">Compute Savings</div>
                      <div className="font-mono font-bold text-emerald-400 text-sm mt-0.5">{solveResult.compute_savings_pct}%</div>
                    </div>
                  </div>

                  {/* Verified Solution Card */}
                  <div className="p-3.5 rounded-lg bg-emerald-950/20 border border-emerald-500/30 space-y-1 mt-2">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider flex items-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        <span>VERIFIED FINAL ANSWER</span>
                      </span>
                    </div>
                    <div className="font-mono font-bold text-white text-sm mt-1">{solveResult.final_answer}</div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Column: Reasoning Step Tree & Inspector */}
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <GitBranch className="w-4 h-4 text-cyan-400" />
                    <h2 className="text-sm font-semibold text-white">Explored Reasoning Tree</h2>
                  </div>
                  <div className="flex items-center gap-3 text-[11px]">
                    <span className="flex items-center gap-1 text-emerald-400 font-mono">
                      <span className="w-2 h-2 rounded-full bg-emerald-500"></span> Accepted
                    </span>
                    <span className="flex items-center gap-1 text-rose-400 font-mono">
                      <span className="w-2 h-2 rounded-full bg-rose-500"></span> Pruned
                    </span>
                    <span className="flex items-center gap-1 text-purple-400 font-mono">
                      <span className="w-2 h-2 rounded-full bg-purple-500"></span> Terminal
                    </span>
                  </div>
                </div>

                {!solveResult ? (
                  <div className="p-12 text-center text-slate-500 text-xs italic border border-dashed border-slate-800 rounded-xl">
                    Configure parameters and click Execute Verifiable Search to build the verified reasoning tree.
                  </div>
                ) : (
                  <div className="space-y-3 max-h-[480px] overflow-y-auto pr-1">
                    {solveResult.tree_nodes.map((node) => {
                      const isSelected = selectedNode?.id === node.id;
                      const isOptimal = solveResult.optimal_trajectory.some((opt) => opt.id === node.id);

                      return (
                        <div
                          key={node.id}
                          onClick={() => setSelectedNode(node)}
                          style={{ marginLeft: `${node.depth * 20}px` }}
                          className={`p-3 rounded-lg border cursor-pointer transition-all text-xs space-y-1.5 ${
                            isSelected
                              ? 'ring-2 ring-cyan-500 bg-slate-900 border-cyan-500/80 shadow-md'
                              : node.status === 'pruned'
                              ? 'bg-rose-950/20 border-rose-500/30 text-slate-300 hover:border-rose-500/60'
                              : node.status === 'terminal'
                              ? 'bg-purple-950/20 border-purple-500/40 text-slate-200 hover:border-purple-500/70'
                              : 'bg-emerald-950/20 border-emerald-500/40 text-slate-200 hover:border-emerald-500/70'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              {node.status === 'pruned' ? (
                                <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />
                              ) : node.status === 'terminal' ? (
                                <CheckCircle2 className="w-3.5 h-3.5 text-purple-400 shrink-0" />
                              ) : (
                                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
                              )}
                              <span className="font-mono text-[10px] text-slate-400">DEPTH {node.depth}</span>
                              {isOptimal && (
                                <span className="text-[9px] bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 px-1.5 py-0.2 rounded font-mono font-bold">
                                  OPTIMAL PATH
                                </span>
                              )}
                            </div>

                            <div className="flex items-center gap-2 font-mono text-[11px]">
                              <span className="text-slate-400">PRM:</span>
                              <span
                                className={`font-bold ${
                                  node.step_score >= pruneThreshold ? 'text-emerald-400' : 'text-rose-400'
                                }`}
                              >
                                {node.step_score.toFixed(2)}
                              </span>
                            </div>
                          </div>

                          <p className="font-mono text-[11px] text-slate-200 pl-5">{node.content}</p>

                          {node.errors.length > 0 && (
                            <div className="pl-5 text-[10px] text-rose-400 font-mono">
                              Pruned: {node.errors.join('; ')}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>

              {/* Node Inspector Drawer */}
              {selectedNode && (
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-bold text-white uppercase tracking-wider flex items-center gap-1.5">
                      <Activity className="w-4 h-4 text-cyan-400" />
                      <span>Step Detail Inspector ({selectedNode.id})</span>
                    </span>
                    <span
                      className={`font-mono text-[10px] px-2 py-0.5 rounded-full border uppercase ${
                        selectedNode.status === 'pruned'
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                          : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                      }`}
                    >
                      {selectedNode.status}
                    </span>
                  </div>

                  <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono text-xs text-slate-200">
                    {selectedNode.content}
                  </div>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-400">PRM Step Confidence</span>
                      <div className="font-mono font-bold text-emerald-400 text-sm mt-0.5">
                        {selectedNode.step_score.toFixed(3)}
                      </div>
                    </div>
                    <div className="p-2.5 bg-slate-950 rounded-lg border border-slate-800">
                      <span className="text-[10px] text-slate-400">Cumulative Path Confidence</span>
                      <div className="font-mono font-bold text-cyan-400 text-sm mt-0.5">
                        {selectedNode.cumulative_score.toFixed(3)}
                      </div>
                    </div>
                  </div>

                  {selectedNode.rationale && (
                    <div className="text-[11px] text-slate-400 pl-1">
                      <span className="text-slate-300 font-semibold">Verifier Rationale:</span> {selectedNode.rationale}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Sub-Tab 2: Paradigm Benchmark Arena */}
        {activeSubTab === 'benchmark' && (
          <div className="space-y-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-semibold text-white flex items-center gap-2">
                    <BarChart2 className="w-5 h-5 text-emerald-400" />
                    <span>Search & Test-Time Compute Arena</span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Benchmarking Greedy Generation vs Best-of-N (ORM) vs Step-Level Verifiable Search (PRM).
                  </p>
                </div>

                <button
                  onClick={handleRunBenchmark}
                  disabled={benchmarkLoading}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-emerald-600/20"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>{benchmarkLoading ? 'Evaluating Paradigms...' : 'Run Comparative Benchmark'}</span>
                </button>
              </div>

              {benchmarkResult ? (
                <div className="space-y-4">
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400 font-mono">
                          <th className="py-2.5 px-3">METHOD</th>
                          <th className="py-2.5 px-3">ACCURACY</th>
                          <th className="py-2.5 px-3">AVG STEPS GENERATED</th>
                          <th className="py-2.5 px-3">EARLY PRUNE RATE</th>
                          <th className="py-2.5 px-3">COMPUTE SAVINGS</th>
                          <th className="py-2.5 px-3">AVG LATENCY</th>
                        </tr>
                      </thead>
                      <tbody>
                        {benchmarkResult.methods.map((m, idx) => (
                          <tr key={idx} className="border-b border-slate-850 hover:bg-slate-900/40">
                            <td className="py-3 px-3 font-semibold text-white flex items-center gap-2">
                              {m.method === benchmarkResult.winner && (
                                <span className="text-[10px] bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 px-1.5 py-0.2 rounded font-mono font-bold">
                                  WINNER
                                </span>
                              )}
                              <span>{m.method}</span>
                            </td>
                            <td className="py-3 px-3 font-mono text-emerald-400">{(m.accuracy * 100).toFixed(1)}%</td>
                            <td className="py-3 px-3 font-mono text-slate-300">{m.avg_steps_generated}</td>
                            <td className="py-3 px-3 font-mono text-amber-400">{(m.early_prune_rate * 100).toFixed(1)}%</td>
                            <td className="py-3 px-3 font-mono text-emerald-300">+{m.compute_savings_pct}%</td>
                            <td className="py-3 px-3 font-mono text-slate-400">{m.avg_latency_ms} ms</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 space-y-1">
                    <span className="font-bold text-cyan-400">Verification Efficiency Note:</span>
                    <p className="text-slate-400 text-[11px]">{benchmarkResult.verifiable_compute_savings}</p>
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center text-slate-500 text-xs italic border border-dashed border-slate-800 rounded-xl">
                  Click Run Comparative Benchmark to compare test-time search strategies.
                </div>
              )}
            </div>
          </div>
        )}

        {/* Sub-Tab 3: PRM Architecture Theory */}
        {activeSubTab === 'theory' && (
          <div className="space-y-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 space-y-4">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-purple-400" />
                <h2 className="text-base font-semibold text-white">Process vs Outcome Supervision</h2>
              </div>
              <p className="text-xs text-slate-400">
                Understanding test-time compute scaling through Process Reward Models and verifiable tree search.
              </p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs mt-2">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <span className="text-rose-400 font-bold">1. Outcome Reward Models (ORM)</span>
                  <p className="text-slate-400 text-[11px]">
                    Scores only the complete answer at depth T. If Step 1 contains a flaw, the model blindly generates T-1 doomed steps, wasting test-time compute.
                  </p>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <span className="text-cyan-400 font-bold">2. Process Reward Models (PRM)</span>
                  <p className="text-slate-400 text-[11px]">
                    Provides credit assignment at step t. Detects arithmetic mismatches (e.g. 15 * 4 = 55) or logical fallacies immediately, giving granular feedback.
                  </p>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
                  <span className="text-emerald-400 font-bold">3. Early Pruning & Backtracking</span>
                  <p className="text-slate-400 text-[11px]">
                    When a step score drops below tau, the engine discards the subtree and backtracks to the next highest-scoring alternative node on the frontier.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
