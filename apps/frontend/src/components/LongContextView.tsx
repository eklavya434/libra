'use client';

import React, { useState, useEffect } from 'react';
import {
  Compass,
  Play,
  Layers,
  Sparkles,
  CheckCircle2,
  XCircle,
  Clock,
  Zap,
  TrendingDown,
  Info,
  Sliders,
  Maximize2,
  FileText,
} from 'lucide-react';

interface NeedleGridResult {
  context_length: number;
  depth_percent: number;
  is_correct: boolean;
  score: number;
  latency_ms: number;
  retrieved_text: string;
}

interface CompactionStats {
  cached_tokens: number;
  total_tokens_seen: number;
  evicted_tokens: number;
  max_budget: number;
  n_sink: number;
  n_recent: number;
  n_heavy: number;
  memory_bytes: number;
  uncompressed_bytes: number;
  memory_savings_pct: number;
}

interface TokenSample {
  index: number;
  type: 'sink' | 'recent' | 'heavy' | 'evicted';
  label: string;
}

export default function LongContextView() {
  const [activeTab, setActiveTab] = useState<'niah' | 'compaction' | 'multi'>('niah');

  // NIAH State
  const [model, setModel] = useState<string>('mock');
  const [needle, setNeedle] = useState<string>('The secret access code to the vault is 849204.');
  const [targetKey, setTargetKey] = useState<string>('849204');
  const [prompt, setPrompt] = useState<string>('What is the secret access code to the vault?');
  const [lengths, setLengths] = useState<number[]>([300, 600, 1000]);
  const [depths, setDepths] = useState<number[]>([0.1, 0.3, 0.5, 0.7, 0.9]);
  const [gridResults, setGridResults] = useState<NeedleGridResult[]>([]);
  const [selectedCell, setSelectedCell] = useState<NeedleGridResult | null>(null);
  const [isEvaluating, setIsEvaluating] = useState<boolean>(false);
  const [accuracy, setAccuracy] = useState<number | null>(null);
  const [avgLatency, setAvgLatency] = useState<number | null>(null);

  // Compaction Simulation State
  const [seqLength, setSeqLength] = useState<number>(1024);
  const [budget, setBudget] = useState<number>(256);
  const [nSink, setNSink] = useState<number>(4);
  const [nRecent, setNRecent] = useState<number>(64);
  const [compactionStats, setCompactionStats] = useState<CompactionStats | null>(null);
  const [tokenSamples, setTokenSamples] = useState<TokenSample[]>([]);
  const [isSimulating, setIsSimulating] = useState<boolean>(false);

  // Multi-Needle State
  const [multiLength, setMultiLength] = useState<number>(600);
  const [multiResult, setMultiResult] = useState<any>(null);
  const [isMultiRunning, setIsMultiRunning] = useState<boolean>(false);

  // Run NIAH Grid
  const runNiahGrid = async () => {
    setIsEvaluating(true);
    try {
      const res = await fetch('/api/v1/long_context/evaluate/needle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          needle,
          target_key: targetKey,
          retrieval_prompt: prompt,
          context_lengths: lengths,
          depth_fractions: depths,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setGridResults(data.results);
        setAccuracy(data.accuracy_percent);
        setAvgLatency(data.average_latency_ms);
        if (data.results.length > 0) {
          setSelectedCell(data.results[0]);
        }
      }
    } catch {
      // Offline fallback
    } finally {
      setIsEvaluating(false);
    }
  };

  // Run Compaction Simulation
  const runCompactionSim = async () => {
    setIsSimulating(true);
    try {
      const res = await fetch('/api/v1/long_context/compaction/simulate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sequence_length: seqLength,
          max_budget: budget,
          n_sink: nSink,
          n_recent: nRecent,
          n_layers: 4,
          n_kv_heads: 4,
          head_dim: 64,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setCompactionStats(data.final_stats);
        setTokenSamples(data.token_sample);
      }
    } catch {
      // Offline fallback
    } finally {
      setIsSimulating(false);
    }
  };

  // Run Multi-Needle Test
  const runMultiNeedle = async () => {
    setIsMultiRunning(true);
    try {
      const res = await fetch('/api/v1/long_context/evaluate/multi_needle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model,
          context_length: multiLength,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setMultiResult(data);
      }
    } catch {
      // Offline fallback
    } finally {
      setIsMultiRunning(false);
    }
  };

  // Initial load
  useEffect(() => {
    runNiahGrid();
    runCompactionSim();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Top Header */}
      <header className="flex flex-wrap items-center justify-between gap-4 px-6 py-4 border-b border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-teal-500/10 border border-teal-500/30 text-teal-400">
            <Compass className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold text-white tracking-wide">
                Long-Context & Attention Compaction Lab
              </h1>
              <span className="px-2 py-0.5 text-[10px] font-mono bg-teal-500/10 text-teal-400 border border-teal-500/30 rounded-full">
                Phase 42
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Needle-In-A-Haystack (NIAH) Heatmaps & H2O/StreamingLLM KV Eviction
            </p>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-1 bg-slate-900 p-1 rounded-lg border border-slate-800">
          <button
            onClick={() => setActiveTab('niah')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'niah'
                ? 'bg-teal-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            2D NIAH Heatmap
          </button>
          <button
            onClick={() => setActiveTab('compaction')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'compaction'
                ? 'bg-teal-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            KV Cache Compaction
          </button>
          <button
            onClick={() => setActiveTab('multi')}
            className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
              activeTab === 'multi'
                ? 'bg-teal-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            Multi-Needle Recall
          </button>
        </div>
      </header>

      {/* Main Workspace Body */}
      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {/* TAB 1: 2D NIAH HEATMAP */}
        {activeTab === 'niah' && (
          <div className="space-y-6">
            {/* Top Telemetry Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                  <span>Overall Accuracy</span>
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                </div>
                <div className="text-2xl font-bold font-mono text-emerald-400">
                  {accuracy !== null ? `${accuracy}%` : '...'}
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Exact associative recall across all trials</p>
              </div>

              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                  <span>Average Latency</span>
                  <Clock className="w-4 h-4 text-cyan-400" />
                </div>
                <div className="text-2xl font-bold font-mono text-cyan-300">
                  {avgLatency !== null ? `${avgLatency} ms` : '...'}
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Mean CPU inference response time</p>
              </div>

              <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                <div className="flex items-center justify-between text-slate-400 text-xs mb-1">
                  <span>Test Matrix Grid</span>
                  <Layers className="w-4 h-4 text-teal-400" />
                </div>
                <div className="text-2xl font-bold font-mono text-teal-300">
                  {lengths.length} × {depths.length} ({lengths.length * depths.length} cells)
                </div>
                <p className="text-[11px] text-slate-400 mt-1">Context Lengths × Depth Percentiles</p>
              </div>
            </div>

            {/* Heatmap & Inspector Split */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Heatmap Matrix Canvas */}
              <div className="lg:col-span-2 p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
                <div className="flex items-center justify-between">
                  <h2 className="text-sm font-semibold text-slate-200">
                    Needle Retrieval Accuracy Heatmap
                  </h2>
                  <button
                    onClick={runNiahGrid}
                    disabled={isEvaluating}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-medium transition-colors shadow-sm disabled:opacity-50"
                  >
                    <Play className="w-3.5 h-3.5 fill-current" />
                    <span>{isEvaluating ? 'Evaluating...' : 'Run Benchmark'}</span>
                  </button>
                </div>

                {/* The 2D Grid Table */}
                <div className="overflow-x-auto border border-slate-800/80 rounded-xl bg-slate-950 p-4">
                  <table className="w-full text-center border-collapse">
                    <thead>
                      <tr>
                        <th className="p-2 text-xs font-mono text-slate-400 border-b border-slate-800">
                          Depth \ Length
                        </th>
                        {lengths.map((len) => (
                          <th
                            key={len}
                            className="p-2 text-xs font-mono text-slate-300 border-b border-slate-800"
                          >
                            {len} words
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {depths.map((d) => {
                        const depthPct = Math.round(d * 100);
                        return (
                          <tr key={depthPct}>
                            <td className="p-2 text-xs font-mono text-slate-400 border-r border-slate-800 font-semibold">
                              {depthPct}%
                            </td>
                            {lengths.map((len) => {
                              const cell = gridResults.find(
                                (r) => r.context_length === len && Math.round(r.depth_percent) === depthPct
                              );
                              const isSelected =
                                selectedCell &&
                                selectedCell.context_length === len &&
                                Math.round(selectedCell.depth_percent) === depthPct;

                              const bg = cell
                                ? cell.is_correct
                                  ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 hover:bg-emerald-500/30'
                                  : 'bg-rose-500/20 text-rose-300 border-rose-500/40 hover:bg-rose-500/30'
                                : 'bg-slate-900/40 text-slate-400 border-slate-800';

                              return (
                                <td key={len} className="p-1.5">
                                  <button
                                    onClick={() => cell && setSelectedCell(cell)}
                                    className={`w-full py-3 px-2 rounded-lg border text-xs font-mono font-semibold transition-all ${bg} ${
                                      isSelected ? 'ring-2 ring-teal-400 shadow-md' : ''
                                    }`}
                                  >
                                    {cell ? (cell.is_correct ? '100%' : '0%') : '...'}
                                  </button>
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="flex items-center gap-4 text-[11px] text-slate-400 justify-center">
                  <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded bg-emerald-500/30 border border-emerald-500" />
                    <span>Exact Recall (100%)</span>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <div className="w-3 h-3 rounded bg-rose-500/30 border border-rose-500" />
                    <span>Failed / Distracted (0%)</span>
                  </div>
                </div>
              </div>

              {/* Cell Inspector Drawer */}
              <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 flex flex-col justify-between">
                <div>
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
                    <Info className="w-4 h-4 text-teal-400" />
                    Trial Inspector
                  </h3>

                  {selectedCell ? (
                    <div className="space-y-3 text-xs">
                      <div className="p-3 rounded-lg border border-slate-800 bg-slate-950 space-y-1.5">
                        <div className="flex justify-between text-slate-400">
                          <span>Context Length:</span>
                          <span className="font-mono text-slate-200">{selectedCell.context_length} words</span>
                        </div>
                        <div className="flex justify-between text-slate-400">
                          <span>Insertion Depth:</span>
                          <span className="font-mono text-slate-200">{selectedCell.depth_percent}%</span>
                        </div>
                        <div className="flex justify-between text-slate-400">
                          <span>Outcome:</span>
                          <span
                            className={`font-mono font-semibold ${
                              selectedCell.is_correct ? 'text-emerald-400' : 'text-rose-400'
                            }`}
                          >
                            {selectedCell.is_correct ? 'PASSED (Exact Match)' : 'FAILED'}
                          </span>
                        </div>
                        <div className="flex justify-between text-slate-400">
                          <span>Latency:</span>
                          <span className="font-mono text-cyan-300">{selectedCell.latency_ms} ms</span>
                        </div>
                      </div>

                      <div>
                        <span className="text-slate-400 block mb-1">Target Key:</span>
                        <div className="font-mono text-teal-300 bg-slate-950 p-2 rounded border border-slate-800">
                          {targetKey}
                        </div>
                      </div>

                      <div>
                        <span className="text-slate-400 block mb-1">Model Completion:</span>
                        <div className="font-mono text-slate-200 bg-slate-950 p-2.5 rounded border border-slate-800 text-[11px] leading-relaxed max-h-40 overflow-y-auto">
                          {selectedCell.retrieved_text}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-12 text-slate-400 text-xs">
                      Click any heatmap cell to inspect prompt and completion details.
                    </div>
                  )}
                </div>

                <div className="mt-4 pt-4 border-t border-slate-800 text-[11px] text-slate-400">
                  Evaluates whether the language model can locate and extract factual targets embedded within distracting technical texts.
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: KV CACHE COMPACTION */}
        {activeTab === 'compaction' && (
          <div className="space-y-6">
            {/* Control Sliders & Setup */}
            <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-teal-400" />
                  <h2 className="text-sm font-semibold text-slate-200">
                    Compacted KV Cache Budget Parameters
                  </h2>
                </div>
                <button
                  onClick={runCompactionSim}
                  disabled={isSimulating}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-medium transition-colors shadow-sm disabled:opacity-50"
                >
                  <Zap className="w-3.5 h-3.5 fill-current" />
                  <span>{isSimulating ? 'Simulating...' : 'Re-Simulate'}</span>
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
                <div>
                  <label className="text-xs text-slate-400 block mb-1">
                    Sequence Length: <span className="font-mono text-teal-300">{seqLength}</span>
                  </label>
                  <input
                    type="range"
                    min="128"
                    max="4096"
                    step="64"
                    value={seqLength}
                    onChange={(e) => setSeqLength(Number(e.target.value))}
                    className="w-full accent-teal-500"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">
                    Max KV Budget (B): <span className="font-mono text-teal-300">{budget}</span>
                  </label>
                  <input
                    type="range"
                    min="64"
                    max="512"
                    step="32"
                    value={budget}
                    onChange={(e) => setBudget(Number(e.target.value))}
                    className="w-full accent-teal-500"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">
                    Attention Sinks (N_sink): <span className="font-mono text-blue-400">{nSink}</span>
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="16"
                    value={nSink}
                    onChange={(e) => setNSink(Number(e.target.value))}
                    className="w-full accent-blue-500"
                  />
                </div>

                <div>
                  <label className="text-xs text-slate-400 block mb-1">
                    Recent Window (N_recent): <span className="font-mono text-emerald-400">{nRecent}</span>
                  </label>
                  <input
                    type="range"
                    min="16"
                    max="128"
                    step="8"
                    value={nRecent}
                    onChange={(e) => setNRecent(Number(e.target.value))}
                    className="w-full accent-emerald-500"
                  />
                </div>
              </div>
            </div>

            {/* Compaction Telemetry Cards */}
            {compactionStats && (
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                  <span className="text-xs text-slate-400 block mb-1">Memory Savings</span>
                  <div className="text-2xl font-bold font-mono text-teal-400 flex items-center gap-1.5">
                    <TrendingDown className="w-5 h-5 text-teal-400" />
                    <span>{compactionStats.memory_savings_pct}%</span>
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1">RAM reduction over naive KV cache</p>
                </div>

                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                  <span className="text-xs text-slate-400 block mb-1">Active Cached Tokens</span>
                  <div className="text-2xl font-bold font-mono text-slate-100">
                    {compactionStats.cached_tokens} / {compactionStats.max_budget}
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1">Bounded memory budget</p>
                </div>

                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                  <span className="text-xs text-slate-400 block mb-1">Evicted Tokens</span>
                  <div className="text-2xl font-bold font-mono text-amber-400">
                    {compactionStats.evicted_tokens}
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1">Transient tokens dropped</p>
                </div>

                <div className="p-4 rounded-xl border border-slate-800 bg-slate-900/40">
                  <span className="text-xs text-slate-400 block mb-1">Compacted KV RAM</span>
                  <div className="text-2xl font-bold font-mono text-indigo-300">
                    {(compactionStats.memory_bytes / 1024).toFixed(1)} KB
                  </div>
                  <p className="text-[11px] text-slate-400 mt-1">
                    vs {(compactionStats.uncompressed_bytes / 1024).toFixed(1)} KB full
                  </p>
                </div>
              </div>
            )}

            {/* Visual Token Tape */}
            <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                  KV Cache Token Tape & Eviction Map (Sample of 128 positions)
                </h3>
                <div className="flex items-center gap-3 text-[10px]">
                  <span className="flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-blue-500" /> Sinks ({nSink})
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-purple-500" /> Heavy Hitters (H2O)
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500" /> Recent Window ({nRecent})
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2.5 h-2.5 rounded-full bg-slate-700 border border-slate-600" /> Evicted
                  </span>
                </div>
              </div>

              {/* Grid of Tokens */}
              <div className="grid grid-cols-16 sm:grid-cols-32 gap-1.5 p-3 rounded-lg border border-slate-800 bg-slate-950">
                {tokenSamples.map((tok) => {
                  let color = 'bg-slate-800/60 border-slate-700/50 text-slate-400';
                  if (tok.type === 'sink') {
                    color = 'bg-blue-600/30 border-blue-500 text-blue-300';
                  } else if (tok.type === 'recent') {
                    color = 'bg-emerald-600/30 border-emerald-500 text-emerald-300';
                  } else if (tok.type === 'heavy') {
                    color = 'bg-purple-600/30 border-purple-500 text-purple-300';
                  }

                  return (
                    <div
                      key={tok.index}
                      title={`Token #${tok.index}: ${tok.label}`}
                      className={`h-7 rounded flex items-center justify-center text-[9px] font-mono border cursor-pointer hover:scale-110 transition-transform ${color}`}
                    >
                      {tok.index}
                    </div>
                  );
                })}
              </div>

              <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-800 text-xs text-slate-400 leading-relaxed">
                <strong className="text-slate-200">How It Works: </strong>
                The <strong>StreamingLLM</strong> policy ensures that initial attention sinks are never evicted, maintaining attention softmax stability. The <strong>H2O (Heavy-Hitter Oracle)</strong> policy tracks cumulative attention scores to keep high-impact prompt tokens while dropping transient filler, bounding cache size to <span className="font-mono text-teal-300">{budget} tokens</span> indefinitely.
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: MULTI-NEEDLE RECALL */}
        {activeTab === 'multi' && (
          <div className="space-y-6">
            <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-sm font-semibold text-slate-200">
                    Multi-Needle Associative Recall Benchmark
                  </h2>
                  <p className="text-xs text-slate-400 mt-0.5">
                    Evaluates cross-key reasoning across multiple needles distributed at different document depths
                  </p>
                </div>
                <button
                  onClick={runMultiNeedle}
                  disabled={isMultiRunning}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-teal-600 hover:bg-teal-500 text-white text-xs font-medium transition-colors shadow-sm disabled:opacity-50"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{isMultiRunning ? 'Running...' : 'Run Multi-Needle Test'}</span>
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
                <div className="p-3 rounded-lg border border-slate-800 bg-slate-950">
                  <div className="text-slate-400 text-xs">Needle 1 (Depth: 20%)</div>
                  <div className="font-mono text-teal-300 text-xs font-semibold mt-1">Alpha-77</div>
                  <div className="text-[11px] text-slate-400 mt-1">Primary launch silo code</div>
                </div>

                <div className="p-3 rounded-lg border border-slate-800 bg-slate-950">
                  <div className="text-slate-400 text-xs">Needle 2 (Depth: 50%)</div>
                  <div className="font-mono text-teal-300 text-xs font-semibold mt-1">Falcon</div>
                  <div className="text-[11px] text-slate-400 mt-1">Mission commander callsign</div>
                </div>

                <div className="p-3 rounded-lg border border-slate-800 bg-slate-950">
                  <div className="text-slate-400 text-xs">Needle 3 (Depth: 85%)</div>
                  <div className="font-mono text-teal-300 text-xs font-semibold mt-1">Omega-Zero</div>
                  <div className="text-[11px] text-slate-400 mt-1">Emergency abort sequence key</div>
                </div>
              </div>
            </div>

            {/* Results display */}
            {multiResult && (
              <div className="p-5 rounded-xl border border-slate-800 bg-slate-900/40 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                    Evaluation Result
                  </h3>
                  <div className="flex items-center gap-2">
                    <span
                      className={`text-xs font-mono font-semibold px-2 py-0.5 rounded ${
                        multiResult.all_correct
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                      }`}
                    >
                      Score: {(multiResult.partial_score * 100).toFixed(0)}%
                    </span>
                    <span className="text-xs text-slate-400 font-mono">
                      {multiResult.latency_ms} ms
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="p-3 rounded-lg border border-slate-800 bg-slate-950 space-y-2">
                    <span className="text-xs text-slate-400 font-medium">Found Keys:</span>
                    <div className="flex flex-wrap gap-1.5">
                      {multiResult.found_keys.map((k: string) => (
                        <span
                          key={k}
                          className="px-2 py-0.5 rounded text-[11px] font-mono bg-emerald-500/20 text-emerald-300 border border-emerald-500/40"
                        >
                          ✓ {k}
                        </span>
                      ))}
                    </div>
                  </div>

                  <div className="p-3 rounded-lg border border-slate-800 bg-slate-950 space-y-2">
                    <span className="text-xs text-slate-400 font-medium">Missing Keys:</span>
                    <div className="flex flex-wrap gap-1.5">
                      {multiResult.missing_keys.length === 0 ? (
                        <span className="text-xs text-slate-400 italic">None (All captured)</span>
                      ) : (
                        multiResult.missing_keys.map((k: string) => (
                          <span
                            key={k}
                            className="px-2 py-0.5 rounded text-[11px] font-mono bg-rose-500/20 text-rose-300 border border-rose-500/40"
                          >
                            ✗ {k}
                          </span>
                        ))
                      )}
                    </div>
                  </div>
                </div>

                <div className="p-3 rounded-lg border border-slate-800 bg-slate-950">
                  <span className="text-xs text-slate-400 block mb-1">Model Synthesis:</span>
                  <div className="font-mono text-xs text-slate-200 leading-relaxed">
                    {multiResult.retrieved_text}
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
