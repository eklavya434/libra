'use client';

import React, { useState, useEffect } from 'react';
import {
  FastForward,
  Zap,
  CheckCircle2,
  XCircle,
  Play,
  RotateCcw,
  BarChart2,
  Sliders,
  Sparkles,
  ArrowRight,
  TrendingUp,
  Cpu,
  Layers,
  Activity,
} from 'lucide-react';

interface Preset {
  id: string;
  name: string;
  description: string;
  num_heads: number;
  decay: number;
  temperature: number;
}

interface StepDetail {
  step: number;
  accepted_count: number;
  drafted_tokens_str: string[];
  accepted_tokens_str: string[];
  acceptance_mask: boolean[];
}

interface GenerateResponse {
  prompt: string;
  generated_text: string;
  generated_tokens_count: number;
  total_forward_passes: number;
  speedup_ratio: number;
  avg_accepted_per_step: number;
  tokens_per_second: number;
  elapsed_seconds: number;
  steps: StepDetail[];
}

interface BenchmarkData {
  benchmark: {
    test_prompts_count: number;
    tokens_generated_per_prompt: number;
    autoregressive_time_ms: number;
    medusa_time_ms: number;
    autoregressive_tokens_per_sec: number;
    medusa_tokens_per_sec: number;
    speedup_ratio: number;
    avg_accepted_tokens_per_step: number;
    forward_pass_reduction_pct: number;
  };
  head_accuracies: {
    num_heads: number;
    per_head_accuracy_pct: number[];
    per_head_eval_tokens: number[];
  };
}

export default function MedusaView() {
  const [activeSubTab, setActiveSubTab] = useState<'playground' | 'benchmark' | 'training'>('playground');
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('medusa_3heads');

  // Playground State
  const [prompt, setPrompt] = useState<string>(
    'Speculative decoding accelerates language models by verifying multiple candidates.'
  );
  const [maxTokens, setMaxTokens] = useState<number>(20);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [genResult, setGenResult] = useState<GenerateResponse | null>(null);

  // Benchmark State
  const [isBenchmarking, setIsBenchmarking] = useState<boolean>(false);
  const [benchData, setBenchData] = useState<BenchmarkData | null>(null);

  // Training State
  const [trainSteps, setTrainSteps] = useState<number>(15);
  const [trainDecay, setTrainDecay] = useState<number>(0.8);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainData, setTrainData] = useState<{
    history: Array<{
      step: number;
      total_loss: number;
      per_head_losses: number[];
    }>;
    initial_loss: number;
    final_loss: number;
    loss_reduction_pct: number;
    num_heads: number;
  } | null>(null);

  // Load Presets
  useEffect(() => {
    fetch('/api/v1/medusa/presets')
      .then((res) => res.json())
      .then((data) => {
        if (data.presets) setPresets(data.presets);
      })
      .catch(() => {});
  }, []);

  // Initial Run
  useEffect(() => {
    handleGenerate();
  }, []);

  const handleGenerate = async () => {
    setIsGenerating(true);
    try {
      const res = await fetch('/api/v1/medusa/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_new_tokens: maxTokens,
          temperature: 0.0,
        }),
      });
      const data = await res.json();
      setGenResult(data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsGenerating(false);
    }
  };

  const handleRunBenchmark = async () => {
    setIsBenchmarking(true);
    try {
      const res = await fetch('/api/v1/medusa/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_new_tokens: 15 }),
      });
      const data = await res.json();
      setBenchData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsBenchmarking(false);
    }
  };

  const handleTrain = async () => {
    setIsTraining(true);
    try {
      const res = await fetch('/api/v1/medusa/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          steps: trainSteps,
          decay: trainDecay,
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

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 text-slate-100 overflow-y-auto">
      {/* Header */}
      <div className="border-b border-slate-800/80 bg-slate-900/60 px-6 py-4 backdrop-blur-sm sticky top-0 z-10">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center text-white shadow-lg shadow-cyan-500/20">
              <FastForward className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-semibold text-slate-100 tracking-tight">
                  Medusa Multi-Head Speculative Lab
                </h1>
                <span className="text-[10px] font-mono uppercase bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 px-2 py-0.5 rounded-full">
                  Phase 46
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Parallel multi-token drafting without auxiliary draft models, prefix verification, and CPU throughput acceleration
              </p>
            </div>
          </div>

          {/* Sub-tab Switcher */}
          <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-1 text-xs">
            <button
              onClick={() => setActiveSubTab('playground')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeSubTab === 'playground'
                  ? 'bg-cyan-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              <span>Speculative Sandbox</span>
            </button>
            <button
              onClick={() => {
                setActiveSubTab('benchmark');
                if (!benchData) handleRunBenchmark();
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeSubTab === 'benchmark'
                  ? 'bg-cyan-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <BarChart2 className="w-3.5 h-3.5" />
              <span>Speedup Benchmark</span>
            </button>
            <button
              onClick={() => setActiveSubTab('training')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeSubTab === 'training'
                  ? 'bg-cyan-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Head Training</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Container */}
      <div className="p-6 space-y-6 max-w-7xl mx-auto w-full">
        {/* Speedup & Acceptance Overview Bar */}
        {genResult && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400">Speculative Speedup Factor</span>
              <div className="text-xl font-mono font-bold text-cyan-400 mt-1">
                {genResult.speedup_ratio}x Faster
              </div>
              <span className="text-[10px] text-slate-400 font-mono">
                {genResult.total_forward_passes} passes for {genResult.generated_tokens_count} tokens
              </span>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400">Avg Tokens per Forward Pass</span>
              <div className="text-xl font-mono font-bold text-emerald-400 mt-1">
                {genResult.avg_accepted_per_step} Tokens
              </div>
              <span className="text-[10px] text-emerald-400 font-mono">
                E[alpha] acceptance rate
              </span>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400">Generation Velocity</span>
              <div className="text-xl font-mono font-bold text-slate-200 mt-1">
                {genResult.tokens_per_second} tok/sec
              </div>
              <span className="text-[10px] text-slate-400 font-mono">
                {genResult.elapsed_seconds}s total latency
              </span>
            </div>

            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
              <span className="text-xs text-slate-400">Forward Passes Saved</span>
              <div className="text-xl font-mono font-bold text-amber-400 mt-1">
                {Math.round(
                  (1.0 - genResult.total_forward_passes / genResult.generated_tokens_count) * 100
                )}%
              </div>
              <span className="text-[10px] text-slate-400 font-mono">
                Compute reduction vs pure AR
              </span>
            </div>
          </div>
        )}

        {/* SUB-TAB 1: PLAYGROUND & VERIFICATION TAPE */}
        {activeSubTab === 'playground' && (
          <div className="space-y-6">
            {/* Prompt & Presets Bar */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-3">
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Zap className="w-4 h-4 text-cyan-400" />
                  Speculative Generation Sandbox
                </h2>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs text-slate-400 mr-1">Presets:</span>
                  {presets.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => {
                        setSelectedPresetId(p.id);
                        if (p.id === 'medusa_2heads') {
                          setMaxTokens(15);
                        } else {
                          setMaxTokens(20);
                        }
                      }}
                      className={`px-2.5 py-1 rounded text-xs transition-all ${
                        selectedPresetId === p.id
                          ? 'bg-cyan-600/30 text-cyan-200 border border-cyan-500/50'
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
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
                  placeholder="Type an input prompt..."
                />
                <button
                  onClick={handleGenerate}
                  disabled={isGenerating}
                  className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-cyan-900/50 text-white px-5 py-2 rounded-lg text-xs font-medium flex items-center gap-1.5 shadow-md shadow-cyan-600/20 cursor-pointer"
                >
                  {isGenerating ? (
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Play className="w-3.5 h-3.5 fill-current" />
                  )}
                  <span>Speculate</span>
                </button>
              </div>

              <div className="flex items-center justify-between pt-1">
                <div className="flex items-center gap-3 text-xs text-slate-400">
                  <span>Tokens to generate: {maxTokens}</span>
                  <input
                    type="range"
                    min="10"
                    max="40"
                    step="5"
                    value={maxTokens}
                    onChange={(e) => setMaxTokens(parseInt(e.target.value))}
                    className="w-32 accent-cyan-500 cursor-pointer"
                  />
                </div>
                <span className="text-[10px] text-slate-400 font-mono">
                  Pure Local CPU (100% Free / Zero External Draft Model)
                </span>
              </div>
            </div>

            {/* Generated Text Display */}
            {genResult && (
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 space-y-4">
                <div>
                  <h3 className="text-xs font-semibold text-slate-200 mb-1.5 flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-cyan-400" />
                    Completed Autoregressive Sequence
                  </h3>
                  <div className="p-3 bg-slate-950/80 rounded-lg border border-slate-850 font-mono text-xs text-slate-200 leading-relaxed">
                    <span className="text-slate-400">{genResult.prompt}</span>
                    <span className="text-cyan-300 font-semibold">{genResult.generated_text}</span>
                  </div>
                </div>

                {/* Step-by-Step Verification Tape */}
                <div className="space-y-3 pt-2">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                      <FastForward className="w-4 h-4 text-cyan-400" />
                      Speculative Verification Tape (Accepted vs Rejected Candidates)
                    </h3>
                    <span className="text-[11px] text-slate-400 font-mono">
                      {genResult.steps.length} Steps Executed
                    </span>
                  </div>

                  <div className="space-y-2 max-h-96 overflow-y-auto pr-1">
                    {genResult.steps.map((step) => (
                      <div
                        key={step.step}
                        className="bg-slate-950 p-3 rounded-lg border border-slate-850 flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs"
                      >
                        <div className="flex items-center gap-2.5">
                          <span className="w-6 h-6 rounded bg-slate-900 flex items-center justify-center font-mono font-bold text-slate-400 text-[11px]">
                            #{step.step}
                          </span>
                          <span className="text-slate-400 text-[11px]">Proposals:</span>

                          <div className="flex flex-wrap gap-1.5">
                            {step.drafted_tokens_str.map((tokStr, idx) => {
                              const isAccepted = step.acceptance_mask[idx];
                              const isBase = idx === 0;

                              return (
                                <span
                                  key={idx}
                                  className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-mono border ${
                                    isAccepted
                                      ? 'bg-emerald-500/10 border-emerald-500/40 text-emerald-300'
                                      : 'bg-rose-500/10 border-rose-500/40 text-rose-300 line-through opacity-70'
                                  }`}
                                >
                                  {isAccepted ? (
                                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                                  ) : (
                                    <XCircle className="w-3 h-3 text-rose-400" />
                                  )}
                                  <span>{tokStr}</span>
                                  <span className="text-[9px] opacity-70">
                                    {isBase ? '(Base)' : `(H${idx})`}
                                  </span>
                                </span>
                              );
                            })}
                          </div>
                        </div>

                        <div className="flex items-center gap-2 text-xs font-mono">
                          <span className="text-slate-400">Accepted:</span>
                          <span className="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 font-bold">
                            +{step.accepted_count} tokens
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* SUB-TAB 2: SPEEDUP BENCHMARK ARENA */}
        {activeSubTab === 'benchmark' && (
          <div className="space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <BarChart2 className="w-4 h-4 text-cyan-400" />
                  Autoregressive vs Medusa Speculative Benchmark
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Benchmarking single-token sequential generation against Medusa multi-head parallel verification.
                </p>
              </div>
              <button
                onClick={handleRunBenchmark}
                disabled={isBenchmarking}
                className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-cyan-900/50 text-white px-4 py-2 rounded-lg text-xs font-medium flex items-center gap-1.5 shadow-md shadow-cyan-600/20 cursor-pointer"
              >
                {isBenchmarking ? (
                  <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Play className="w-3.5 h-3.5 fill-current" />
                )}
                <span>Run Benchmark</span>
              </button>
            </div>

            {benchData && (
              <div className="space-y-6">
                {/* Comparison Bar */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Baseline Autoregressive */}
                  <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 space-y-4">
                    <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                      <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                        <Cpu className="w-4 h-4" />
                        Standard Autoregressive (Baseline)
                      </span>
                      <span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded font-mono">
                        1 Token / Pass
                      </span>
                    </div>

                    <div className="space-y-3 font-mono text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Latency:</span>
                        <span className="text-slate-200 font-bold">
                          {benchData.benchmark.autoregressive_time_ms} ms
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Throughput:</span>
                        <span className="text-slate-200 font-bold">
                          {benchData.benchmark.autoregressive_tokens_per_sec} tok/sec
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Forward Passes:</span>
                        <span className="text-slate-200">
                          {benchData.benchmark.tokens_generated_per_prompt * benchData.benchmark.test_prompts_count} passes
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Medusa Speculative */}
                  <div className="bg-cyan-950/20 border border-cyan-500/30 rounded-xl p-5 space-y-4">
                    <div className="flex items-center justify-between border-b border-cyan-500/20 pb-3">
                      <span className="text-xs font-semibold text-cyan-300 uppercase tracking-wider flex items-center gap-1.5">
                        <FastForward className="w-4 h-4 text-cyan-400" />
                        Medusa Speculative Decoding
                      </span>
                      <span className="text-[10px] bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 px-1.5 py-0.5 rounded font-mono">
                        Parallel Heads
                      </span>
                    </div>

                    <div className="space-y-3 font-mono text-xs">
                      <div className="flex justify-between">
                        <span className="text-slate-400">Latency:</span>
                        <span className="text-cyan-300 font-bold">
                          {benchData.benchmark.medusa_time_ms} ms
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Throughput:</span>
                        <span className="text-cyan-300 font-bold">
                          {benchData.benchmark.medusa_tokens_per_sec} tok/sec
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Average E[alpha]:</span>
                        <span className="text-emerald-400 font-bold">
                          {benchData.benchmark.avg_accepted_tokens_per_step} tokens / pass
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-slate-400">Speedup:</span>
                        <span className="text-emerald-400 font-bold">
                          {benchData.benchmark.speedup_ratio}x Faster
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Per-Head Accuracy Breakdown */}
                <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 space-y-4">
                  <h3 className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-cyan-400" />
                    Speculative Head Prediction Accuracy (Top-1 Match vs Ground Truth)
                  </h3>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {benchData.head_accuracies.per_head_accuracy_pct.map((acc, hIdx) => (
                      <div
                        key={hIdx}
                        className="bg-slate-950 p-4 rounded-xl border border-slate-850 space-y-2"
                      >
                        <div className="flex justify-between items-center">
                          <span className="text-xs font-semibold text-slate-300 font-mono">
                            Medusa Head H{hIdx + 1}
                          </span>
                          <span className="text-[10px] text-slate-400 font-mono">
                            Predicts t + {hIdx + 2}
                          </span>
                        </div>
                        <div className="text-xl font-mono font-bold text-cyan-300">
                          {acc}%
                        </div>
                        <div className="w-full bg-slate-850 rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-cyan-500 h-full rounded-full transition-all"
                            style={{ width: `${acc}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* SUB-TAB 3: HEAD TRAINING */}
        {activeSubTab === 'training' && (
          <div className="space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="border-b border-slate-800 pb-3">
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Layers className="w-4 h-4 text-cyan-400" />
                  Medusa Speculative Heads Training Engine
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Trains speculative decoding heads while the base language model remains completely frozen. Loss is discounted future cross-entropy: L_medusa = sum_k (decay^k * L_k).
                </p>
              </div>

              {/* Sliders */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-850">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-400 font-medium">Head Loss Discount Factor (decay)</span>
                    <span className="font-mono text-cyan-400 font-semibold">{trainDecay.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="1.0"
                    step="0.05"
                    value={trainDecay}
                    onChange={(e) => setTrainDecay(parseFloat(e.target.value))}
                    className="w-full accent-cyan-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-400 mt-1">Discounts predictions further into future</p>
                </div>

                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-850">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-400 font-medium">Training Steps</span>
                    <span className="font-mono text-cyan-400 font-semibold">{trainSteps}</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="30"
                    step="5"
                    value={trainSteps}
                    onChange={(e) => setTrainSteps(parseInt(e.target.value))}
                    className="w-full accent-cyan-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-400 mt-1">Lightweight CPU optimization passes</p>
                </div>
              </div>

              <div className="flex justify-end pt-2">
                <button
                  onClick={handleTrain}
                  disabled={isTraining}
                  className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-cyan-900/50 text-white px-5 py-2 rounded-lg text-xs font-medium flex items-center gap-2 shadow-md shadow-cyan-600/20 cursor-pointer"
                >
                  {isTraining ? (
                    <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Play className="w-3.5 h-3.5 fill-current" />
                  )}
                  <span>Optimize Speculative Heads</span>
                </button>
              </div>
            </div>

            {/* Training Results */}
            {trainData && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                    <span className="text-xs text-slate-400">Total Loss Delta</span>
                    <div className="flex items-baseline gap-2 mt-1">
                      <span className="text-lg font-mono font-bold text-slate-200">
                        {trainData.initial_loss.toFixed(3)}
                      </span>
                      <ArrowRight className="w-3 h-3 text-slate-400" />
                      <span className="text-lg font-mono font-bold text-emerald-400">
                        {trainData.final_loss.toFixed(3)}
                      </span>
                    </div>
                    <span className="text-[10px] text-emerald-400 font-mono">
                      -{trainData.loss_reduction_pct}% reduction
                    </span>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                    <span className="text-xs text-slate-400">Active Speculative Heads</span>
                    <div className="text-lg font-mono font-bold text-cyan-400 mt-1">
                      {trainData.num_heads} Residual Heads
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">
                      Base model frozen (grad = False)
                    </span>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                    <span className="text-xs text-slate-400">Training Telemetry</span>
                    <div className="text-lg font-mono font-bold text-purple-400 mt-1">
                      {trainData.history.length} Steps
                    </div>
                    <span className="text-[10px] text-slate-400 font-mono">
                      100% Free Local CPU
                    </span>
                  </div>
                </div>

                {/* Training Table */}
                <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 space-y-3">
                  <h3 className="text-xs font-semibold text-slate-200">
                    Per-Head Discounted Loss Telemetry
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="text-[11px] uppercase bg-slate-950/80 text-slate-400 font-mono border-b border-slate-800">
                        <tr>
                          <th className="py-2 px-3">Step</th>
                          <th className="py-2 px-3">Total Medusa Loss</th>
                          <th className="py-2 px-3">Head 1 (t+2)</th>
                          <th className="py-2 px-3">Head 2 (t+3)</th>
                          <th className="py-2 px-3">Head 3 (t+4)</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-850 font-mono text-xs">
                        {trainData.history.map((row) => (
                          <tr key={row.step} className="hover:bg-slate-800/30">
                            <td className="py-2 px-3 font-semibold text-slate-400">#{row.step}</td>
                            <td className="py-2 px-3 text-cyan-300 font-bold">{row.total_loss}</td>
                            <td className="py-2 px-3 text-slate-200">
                              {row.per_head_losses[0] ?? '-'}
                            </td>
                            <td className="py-2 px-3 text-slate-200">
                              {row.per_head_losses[1] ?? '-'}
                            </td>
                            <td className="py-2 px-3 text-slate-200">
                              {row.per_head_losses[2] ?? '-'}
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
