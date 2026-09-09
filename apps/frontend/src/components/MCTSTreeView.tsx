'use client';

import React, { useState, useEffect } from 'react';
import {
  GitBranch,
  Play,
  Sparkles,
  CheckCircle2,
  XCircle,
  Clock,
  Layers,
  Info,
  Sliders,
  Award,
  ArrowRight,
  ShieldCheck,
  AlertTriangle,
} from 'lucide-react';

interface MCTSNodeData {
  id: string;
  parent_id: string | null;
  content: string;
  depth: number;
  visits: number;
  q_value: number;
  prm_score: number;
  is_terminal: boolean;
  is_expanded: boolean;
  num_children: number;
}

interface MCTSResultData {
  prompt: string;
  optimal_path: MCTSNodeData[];
  final_answer: string;
  total_nodes: number;
  total_simulations: number;
  tree_nodes: MCTSNodeData[];
  tree_edges: { source: string; target: string }[];
  best_value: number;
}

interface PRMStepScore {
  step_index: number;
  content: string;
  score: number;
  is_valid: boolean;
  rationale: string;
  detected_errors: string[];
}

interface PRMTraceResult {
  is_valid_trace: boolean;
  first_error_index: number | null;
  mean_step_score: number;
  min_step_score: number;
  steps: PRMStepScore[];
}

interface DPOPair {
  prompt: string;
  chosen: string;
  rejected: string;
  chosen_score: number;
  rejected_score: number;
  reward_margin: number;
  failing_step_index: number | null;
}

export default function MCTSTreeView() {
  const [activeTab, setActiveTab] = useState<'mcts' | 'prm' | 'self_play'>('mcts');

  // MCTS Search State
  const [prompt, setPrompt] = useState<string>(
    'Using numbers [4, 4, 7, 7] and basic arithmetic operations (+, -, *, /), make exactly 24.'
  );
  const [simulations, setSimulations] = useState<number>(15);
  const [branchFactor, setBranchFactor] = useState<number>(3);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [mctsResult, setMctsResult] = useState<MCTSResultData | null>(null);
  const [selectedNode, setSelectedNode] = useState<MCTSNodeData | null>(null);

  // PRM Validator State
  const [prmStepsText, setPrmStepsText] = useState<string>(
    'Step 1: Compute 7 / 7 = 1.\nStep 2: 7 - 1 = 6.\nStep 3: 4 * 6 = 24.'
  );
  const [prmResult, setPrmResult] = useState<PRMTraceResult | null>(null);
  const [isScoringPRM, setIsScoringPRM] = useState<boolean>(false);

  // Self-Play DPO State
  const [dpoPairs, setDpoPairs] = useState<DPOPair[]>([]);
  const [isGeneratingSelfPlay, setIsGeneratingSelfPlay] = useState<boolean>(false);

  // Run MCTS Tree Search
  const runMCTSSearch = async (customPrompt?: string) => {
    setIsSearching(true);
    const targetPrompt = customPrompt || prompt;
    try {
      const res = await fetch('/api/v1/reasoning/mcts/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: targetPrompt,
          n_simulations: simulations,
          branch_factor: branchFactor,
          max_depth: 5,
        }),
      });
      if (res.ok) {
        const data: MCTSResultData = await res.json();
        setMctsResult(data);
        if (data.optimal_path.length > 0) {
          setSelectedNode(data.optimal_path[data.optimal_path.length - 1]);
        }
      }
    } catch {
      // Offline fallback
    } finally {
      setIsSearching(false);
    }
  };

  // Run PRM Scoring
  const runPRMScoring = async () => {
    setIsScoringPRM(true);
    const steps = prmStepsText.split('\n').filter((s) => s.trim().length > 0);
    try {
      const res = await fetch('/api/v1/reasoning/mcts/prm/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ steps }),
      });
      if (res.ok) {
        const data: PRMTraceResult = await res.json();
        setPrmResult(data);
      }
    } catch {
      // Offline fallback
    } finally {
      setIsScoringPRM(false);
    }
  };

  // Run Self-Play DPO Generation
  const runSelfPlay = async () => {
    setIsGeneratingSelfPlay(true);
    try {
      const res = await fetch('/api/v1/reasoning/mcts/self_play/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 3 }),
      });
      if (res.ok) {
        const data: DPOPair[] = await res.json();
        setDpoPairs(data);
      }
    } catch {
      // Offline fallback
    } finally {
      setIsGeneratingSelfPlay(false);
    }
  };

  useEffect(() => {
    runMCTSSearch();
    runPRMScoring();
    runSelfPlay();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Organize nodes by depth for visualization
  const nodesByDepth: Record<number, MCTSNodeData[]> = {};
  if (mctsResult) {
    mctsResult.tree_nodes.forEach((n) => {
      if (!nodesByDepth[n.depth]) {
        nodesByDepth[n.depth] = [];
      }
      nodesByDepth[n.depth].push(n);
    });
  }

  const optimalNodeIds = new Set(mctsResult?.optimal_path.map((n) => n.id) || []);

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Header */}
      <header className="flex flex-wrap items-center justify-between gap-4 px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-violet-500/10 border border-violet-500/30 text-violet-400">
            <GitBranch className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold text-white tracking-wide">
                MCTS Reasoning & Process Reward Model Lab
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-mono bg-violet-500/10 text-violet-400 border border-violet-500/30 rounded-full">
                Phase 43
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Monte Carlo Tree Search (PUCT), Step-Level PRM Verification & Self-Play
            </p>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setActiveTab('mcts')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'mcts'
                ? 'bg-violet-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            MCTS Tree Search
          </button>
          <button
            onClick={() => setActiveTab('prm')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'prm'
                ? 'bg-violet-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Process Reward Model (PRM)
          </button>
          <button
            onClick={() => setActiveTab('self_play')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'self_play'
                ? 'bg-violet-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Self-Play DPO Pairs
          </button>
        </div>
      </header>

      {/* Main Workspace Body */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* TAB 1: MCTS TREE SEARCH */}
        {activeTab === 'mcts' && (
          <div className="space-y-6">
            {/* Controls Bar */}
            <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-violet-400" />
                  <h2 className="text-sm font-semibold text-slate-200">
                    Reasoning Problem & Search Parameters
                  </h2>
                </div>
                <button
                  onClick={() => runMCTSSearch()}
                  disabled={isSearching}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium transition-colors shadow-sm disabled:opacity-50"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{isSearching ? 'Searching...' : 'Run MCTS Search'}</span>
                </button>
              </div>

              <div>
                <label className="text-xs text-slate-400 block mb-1">Reasoning Prompt:</label>
                <input
                  type="text"
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  className="w-full bg-slate-950 text-slate-200 text-xs rounded-lg p-2.5 border border-slate-800 focus:outline-none focus:border-violet-500"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="text-xs text-slate-400 block mb-1">
                    Simulations (Rollouts): <span className="font-mono text-violet-300">{simulations}</span>
                  </label>
                  <input
                    type="range"
                    min="5"
                    max="30"
                    step="1"
                    value={simulations}
                    onChange={(e) => setSimulations(Number(e.target.value))}
                    className="w-full accent-violet-500"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">
                    Branch Factor (k): <span className="font-mono text-violet-300">{branchFactor}</span>
                  </label>
                  <input
                    type="range"
                    min="2"
                    max="4"
                    step="1"
                    value={branchFactor}
                    onChange={(e) => setBranchFactor(Number(e.target.value))}
                    className="w-full accent-violet-500"
                  />
                </div>
              </div>
            </div>

            {/* Optimal Answer Banner */}
            {mctsResult && (
              <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-950/20 flex items-start justify-between gap-4">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <Award className="w-4 h-4 text-emerald-400" />
                    <span className="text-xs font-semibold text-emerald-300 uppercase tracking-wider">
                      MCTS Verified Solution
                    </span>
                  </div>
                  <p className="text-sm font-semibold font-mono text-slate-100">
                    {mctsResult.final_answer}
                  </p>
                  <p className="text-[11px] text-slate-400">
                    Best Q-Value: <span className="text-emerald-400 font-mono">{mctsResult.best_value}</span> across{' '}
                    <span className="text-slate-300 font-mono">{mctsResult.total_nodes} nodes</span> in{' '}
                    <span className="text-slate-300 font-mono">{mctsResult.total_simulations} simulations</span>
                  </p>
                </div>
              </div>
            )}

            {/* Tree Canvas & Step Inspector Split */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Tree Canvas */}
              <div className="lg:col-span-2 p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    Search Tree Graph (Depth Levels)
                  </h3>
                  <div className="flex items-center gap-3 text-[10px] text-slate-400">
                    <span className="flex items-center gap-1">
                      <span className="w-2.5 h-2.5 rounded bg-emerald-500" /> Valid Step (r ≥ 0.70)
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2.5 h-2.5 rounded bg-amber-500" /> Exploring
                    </span>
                    <span className="flex items-center gap-1">
                      <span className="w-2.5 h-2.5 rounded bg-rose-500" /> Flawed / Pruned
                    </span>
                  </div>
                </div>

                {/* Nodes organized by depth */}
                <div className="space-y-5 p-4 rounded-xl border border-slate-800 bg-slate-950 overflow-x-auto">
                  {Object.keys(nodesByDepth)
                    .sort((a, b) => Number(a) - Number(b))
                    .map((depthStr) => {
                      const depth = Number(depthStr);
                      const depthNodes = nodesByDepth[depth];

                      return (
                        <div key={depth} className="space-y-1.5">
                          <div className="text-[10px] font-mono text-slate-500 uppercase tracking-wider">
                            Level {depth} {depth === 0 ? '(Root)' : `(Step ${depth})`}
                          </div>
                          <div className="flex flex-wrap gap-2.5">
                            {depthNodes.map((n) => {
                              const isOptimal = optimalNodeIds.has(n.id);
                              const isSelected = selectedNode?.id === n.id;

                              let color = 'bg-amber-500/10 border-amber-500/30 text-amber-300';
                              if (n.prm_score >= 0.70) {
                                color = 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300';
                              } else if (n.prm_score < 0.55) {
                                color = 'bg-rose-500/10 border-rose-500/30 text-rose-300';
                              }

                              return (
                                <button
                                  key={n.id}
                                  onClick={() => setSelectedNode(n)}
                                  className={`p-2.5 rounded-lg border text-left max-w-xs text-xs font-mono transition-all hover:scale-102 ${color} ${
                                    isOptimal ? 'ring-2 ring-emerald-400 shadow-md shadow-emerald-950/40' : ''
                                  } ${isSelected ? 'border-violet-400 ring-2 ring-violet-500' : ''}`}
                                >
                                  <div className="flex items-center justify-between gap-2 mb-1">
                                    <span className="text-[10px] font-bold text-slate-400">Node #{n.id}</span>
                                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 text-slate-300">
                                      N={n.visits} | Q={n.q_value}
                                    </span>
                                  </div>
                                  <div className="truncate text-[11px] text-slate-200">
                                    {n.content}
                                  </div>
                                  <div className="text-[10px] text-slate-400 mt-1 flex justify-between">
                                    <span>PRM Score:</span>
                                    <span className="font-bold text-slate-200">{(n.prm_score * 100).toFixed(0)}%</span>
                                  </div>
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      );
                    })}
                </div>
              </div>

              {/* Step Inspector Drawer */}
              <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 flex flex-col justify-between">
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
                    <Info className="w-4 h-4 text-violet-400" />
                    Step Inspector
                  </h3>

                  {selectedNode ? (
                    <div className="space-y-3 text-xs">
                      <div className="p-3 rounded-lg border border-slate-800 bg-slate-950 space-y-1.5">
                        <div className="flex justify-between text-slate-400">
                          <span>Node ID:</span>
                          <span className="font-mono text-slate-200">{selectedNode.id}</span>
                        </div>
                        <div className="flex justify-between text-slate-400">
                          <span>Depth:</span>
                          <span className="font-mono text-slate-200">Level {selectedNode.depth}</span>
                        </div>
                        <div className="flex justify-between text-slate-400">
                          <span>Visit Count (N):</span>
                          <span className="font-mono text-cyan-300">{selectedNode.visits}</span>
                        </div>
                        <div className="flex justify-between text-slate-400">
                          <span>Action-Value (Q):</span>
                          <span className="font-mono text-emerald-400">{selectedNode.q_value}</span>
                        </div>
                        <div className="flex justify-between text-slate-400">
                          <span>PRM Step Validity:</span>
                          <span className="font-mono text-violet-300">
                            {(selectedNode.prm_score * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      <div>
                        <span className="text-slate-400 block mb-1">Step Content:</span>
                        <div className="font-mono text-slate-200 bg-slate-950 p-2.5 rounded border border-slate-800 text-xs leading-relaxed">
                          {selectedNode.content}
                        </div>
                      </div>

                      <div>
                        <span className="text-slate-400 block mb-1">Status:</span>
                        <span
                          className={`text-xs px-2 py-0.5 rounded font-mono ${
                            selectedNode.prm_score >= 0.70
                              ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                              : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                          }`}
                        >
                          {selectedNode.prm_score >= 0.70 ? '✓ Verified Valid Step' : '✗ Potential Error Step'}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-slate-400 text-xs">
                      Click any tree node to inspect its PUCT metrics and PRM evaluation.
                    </div>
                  )}
                </div>

                <div className="mt-4 pt-4 border-t border-slate-800 text-[11px] text-slate-400">
                  MCTS guides exploration by balancing high empirical reward $Q(s, a)$ with exploration of less-visited branches via Upper Confidence bounds.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: PRM VALIDATOR */}
        {activeTab === 'prm' && (
          <div className="space-y-6">
            <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-slate-200">
                    Process Reward Model (PRM) Step Verifier
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Evaluates intermediate reasoning steps individually to catch errors early
                  </p>
                </div>
                <button
                  onClick={runPRMScoring}
                  disabled={isScoringPRM}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium transition-colors shadow-sm disabled:opacity-50"
                >
                  <ShieldCheck className="w-3.5 h-3.5" />
                  <span>{isScoringPRM ? 'Scoring...' : 'Verify Reasoning Steps'}</span>
                </button>
              </div>

              <div>
                <label className="text-xs text-slate-400 block mb-1">Enter Reasoning Steps (one per line):</label>
                <textarea
                  value={prmStepsText}
                  onChange={(e) => setPrmStepsText(e.target.value)}
                  rows={4}
                  className="w-full bg-slate-950 text-slate-200 font-mono text-xs rounded-lg p-3 border border-slate-800 focus:outline-none focus:border-violet-500"
                />
              </div>
            </div>

            {/* PRM Score Cards */}
            {prmResult && (
              <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    Step-by-Step Validity Breakdown
                  </h3>
                  <div className="flex items-center gap-3 text-xs font-mono">
                    <span
                      className={`px-2 py-0.5 rounded ${
                        prmResult.is_valid_trace
                          ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                          : 'bg-rose-500/20 text-rose-300 border border-rose-500/40'
                      }`}
                    >
                      {prmResult.is_valid_trace ? '✓ Overall Valid Trace' : '✗ Flawed Trajectory Detected'}
                    </span>
                    <span className="text-slate-400">Mean Score: {prmResult.mean_step_score}</span>
                  </div>
                </div>

                <div className="space-y-3">
                  {prmResult.steps.map((s) => (
                    <div
                      key={s.step_index}
                      className={`p-3.5 rounded-xl border transition-all ${
                        s.is_valid
                          ? 'border-emerald-500/30 bg-emerald-950/10'
                          : 'border-rose-500/40 bg-rose-950/20'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1.5">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-bold text-slate-300">
                            Step {s.step_index + 1}
                          </span>
                          <span
                            className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                              s.is_valid ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'
                            }`}
                          >
                            Score: {(s.score * 100).toFixed(0)}%
                          </span>
                        </div>
                        <span className="text-[11px] text-slate-400">{s.rationale}</span>
                      </div>
                      <div className="font-mono text-xs text-slate-100 bg-slate-950/60 p-2 rounded border border-slate-800/80">
                        {s.content}
                      </div>
                      {s.detected_errors.length > 0 && (
                        <div className="mt-2 text-[11px] text-rose-400 flex items-center gap-1">
                          <AlertTriangle className="w-3 h-3 text-rose-500" />
                          <span>Errors: {s.detected_errors.join(', ')}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* TAB 3: SELF-PLAY DPO PAIRS */}
        {activeTab === 'self_play' && (
          <div className="space-y-6">
            <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-slate-200">
                    Self-Play Trajectory Generator & DPO Pairs
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Synthesizes step-level preference datasets (chosen vs rejected) for continuous self-improvement
                  </p>
                </div>
                <button
                  onClick={runSelfPlay}
                  disabled={isGeneratingSelfPlay}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-violet-600 hover:bg-violet-500 text-white text-xs font-medium transition-colors shadow-sm disabled:opacity-50"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{isGeneratingSelfPlay ? 'Synthesizing...' : 'Generate New Pairs'}</span>
                </button>
              </div>
            </div>

            <div className="space-y-4">
              {dpoPairs.map((pair, idx) => (
                <div key={idx} className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-3">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                    <span className="text-xs font-semibold text-slate-200">Problem #{idx + 1}: {pair.prompt}</span>
                    <span className="text-xs font-mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/30">
                      Margin: +{pair.reward_margin}
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {/* Chosen */}
                    <div className="p-3.5 rounded-lg border border-emerald-500/30 bg-emerald-950/10 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-emerald-300">✓ Chosen Trajectory</span>
                        <span className="font-mono text-emerald-400">Score: {pair.chosen_score}</span>
                      </div>
                      <div className="font-mono text-xs text-slate-200 whitespace-pre-wrap leading-relaxed bg-slate-950/60 p-2.5 rounded border border-slate-800">
                        {pair.chosen}
                      </div>
                    </div>

                    {/* Rejected */}
                    <div className="p-3.5 rounded-lg border border-rose-500/30 bg-rose-950/10 space-y-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="font-semibold text-rose-300">✗ Rejected Trajectory</span>
                        <span className="font-mono text-rose-400">Score: {pair.rejected_score}</span>
                      </div>
                      <div className="font-mono text-xs text-slate-300 whitespace-pre-wrap leading-relaxed bg-slate-950/60 p-2.5 rounded border border-slate-800">
                        {pair.rejected}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
