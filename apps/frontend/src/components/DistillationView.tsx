'use client';

import React, { useState, useEffect } from 'react';
import {
  Shrink,
  Zap,
  Layers,
  Sparkles,
  TrendingDown,
  Gauge,
  Sliders,
  CheckCircle2,
  XCircle,
  Play,
  RotateCcw,
  BarChart2,
  Cpu,
  ArrowRight,
  Flame,
} from 'lucide-react';

interface Preset {
  id: string;
  name: string;
  description: string;
  temperature: number;
  alpha: number;
  hidden_loss_weight: number;
  learning_rate: number;
  steps: number;
}

interface TelemetryItem {
  step: number;
  total_loss: number;
  soft_loss: number;
  hard_loss: number;
  hidden_loss: number;
  kl_div: number;
  top1_agreement: number;
  student_ppl: number;
  teacher_ppl: number;
}

interface CompressionStats {
  teacher_params: number;
  student_params: number;
  compression_ratio: number;
  param_reduction_pct: number;
  teacher_memory_mb: number;
  student_memory_mb: number;
  memory_savings_mb: number;
  teacher_layers: number;
  student_layers: number;
  teacher_d_model: number;
  student_d_model: number;
}

interface TokenProb {
  token_id: number;
  token_str: string;
  prob: number;
}

interface TempAnalysis {
  temperature: number;
  kl_divergence: number;
  js_divergence: number;
  teacher_entropy: number;
  student_entropy: number;
  teacher_top_tokens: TokenProb[];
  student_top_tokens: TokenProb[];
}

export default function DistillationView() {
  const [activeSubTab, setActiveSubTab] = useState<'trainer' | 'dark_knowledge' | 'benchmark'>('trainer');
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('standard_balanced');

  // Training State
  const [temperature, setTemperature] = useState<number>(2.0);
  const [alpha, setAlpha] = useState<number>(0.5);
  const [learningRate, setLearningRate] = useState<number>(0.002);
  const [trainingSteps, setTrainingSteps] = useState<number>(15);
  const [hiddenWeight, setHiddenWeight] = useState<number>(0.0);
  const [customText, setCustomText] = useState<string>(
    'Knowledge distillation transfers soft probability dark knowledge from large teacher to compact student.'
  );

  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [telemetry, setTelemetry] = useState<TelemetryItem[]>([]);
  const [compressionStats, setCompressionStats] = useState<CompressionStats | null>(null);
  const [trainSummary, setTrainSummary] = useState<{
    initialLoss: number;
    finalLoss: number;
    initialAgree: number;
    finalAgree: number;
    improvementPct: number;
  } | null>(null);

  // Dark Knowledge State
  const [darkPrompt, setDarkPrompt] = useState<string>('The capital of France is Paris');
  const [selectedTemp, setSelectedTemp] = useState<number>(2.0);
  const [isLoadingDark, setIsLoadingDark] = useState<boolean>(false);
  const [tempAnalyses, setTempAnalyses] = useState<TempAnalysis[]>([]);

  // Benchmark State
  const [isBenchmarking, setIsBenchmarking] = useState<boolean>(false);
  const [benchmarkData, setBenchmarkData] = useState<{
    benchmark: any;
    compression_stats: CompressionStats;
    sample_predictions: Array<{
      prompt: string;
      teacher_next_token: string;
      student_next_token: string;
      tokens_match: boolean;
    }>;
  } | null>(null);

  // Load Presets
  useEffect(() => {
    fetch('/api/v1/distillation/presets')
      .then((res) => res.json())
      .then((data) => {
        if (data.presets) {
          setPresets(data.presets);
        }
      })
      .catch(() => {});
  }, []);

  const handleApplyPreset = (presetId: string) => {
    setSelectedPresetId(presetId);
    const p = presets.find((item) => item.id === presetId);
    if (p) {
      setTemperature(p.temperature);
      setAlpha(p.alpha);
      setHiddenWeight(p.hidden_loss_weight);
      setLearningRate(p.learning_rate);
      setTrainingSteps(p.steps);
    }
  };

  // Run Training
  const handleTrain = async () => {
    setIsTraining(true);
    try {
      const res = await fetch('/api/v1/distillation/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          temperature,
          alpha,
          lr: learningRate,
          steps: trainingSteps,
          hidden_loss_weight: hiddenWeight,
          custom_text: customText,
        }),
      });
      const data = await res.json();
      if (data.telemetry) {
        setTelemetry(data.telemetry);
        setCompressionStats(data.compression_stats);
        setTrainSummary({
          initialLoss: data.initial_total_loss,
          finalLoss: data.final_total_loss,
          initialAgree: data.initial_agreement,
          finalAgree: data.final_agreement,
          improvementPct: data.agreement_improvement_pct,
        });
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsTraining(false);
    }
  };

  // Run Dark Knowledge Analysis
  const handleAnalyzeDarkKnowledge = async () => {
    setIsLoadingDark(true);
    try {
      const res = await fetch('/api/v1/distillation/soft_labels', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: darkPrompt,
          temperatures: [1.0, 2.0, 5.0, 10.0],
          top_k: 5,
        }),
      });
      const data = await res.json();
      if (data.temperature_analysis) {
        setTempAnalyses(data.temperature_analysis);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoadingDark(false);
    }
  };

  // Run Benchmark
  const handleRunBenchmark = async () => {
    setIsBenchmarking(true);
    try {
      const res = await fetch('/api/v1/distillation/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ runs: 3 }),
      });
      const data = await res.json();
      setBenchmarkData(data);
    } catch (e) {
      console.error(e);
    } finally {
      setIsBenchmarking(false);
    }
  };

  const currentAnalysis =
    tempAnalyses.find((t) => Math.abs(t.temperature - selectedTemp) < 0.1) || tempAnalyses[0];

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 text-slate-100 overflow-y-auto">
      {/* Header */}
      <div className="border-b border-slate-800/80 bg-slate-900/60 px-6 py-4 backdrop-blur-sm sticky top-0 z-10">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-rose-500 to-amber-500 flex items-center justify-center text-white shadow-lg shadow-rose-500/20">
              <Shrink className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-semibold text-slate-100 tracking-tight">
                  Knowledge Distillation & Model Shrinking Lab
                </h1>
                <span className="text-[10px] font-mono uppercase bg-rose-500/20 text-rose-300 border border-rose-500/30 px-2 py-0.5 rounded-full">
                  Phase 44
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Temperature-scaled soft loss (KL Divergence), dark knowledge extraction, and layer-dropping student pruning
              </p>
            </div>
          </div>

          {/* Sub-tab Switcher */}
          <div className="flex items-center bg-slate-950 border border-slate-800 rounded-lg p-1 text-xs">
            <button
              onClick={() => setActiveSubTab('trainer')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeSubTab === 'trainer'
                  ? 'bg-rose-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Zap className="w-3.5 h-3.5" />
              <span>Distillation Trainer</span>
            </button>
            <button
              onClick={() => {
                setActiveSubTab('dark_knowledge');
                if (tempAnalyses.length === 0) handleAnalyzeDarkKnowledge();
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeSubTab === 'dark_knowledge'
                  ? 'bg-rose-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Flame className="w-3.5 h-3.5" />
              <span>Dark Knowledge Explorer</span>
            </button>
            <button
              onClick={() => {
                setActiveSubTab('benchmark');
                if (!benchmarkData) handleRunBenchmark();
              }}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
                activeSubTab === 'benchmark'
                  ? 'bg-rose-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Gauge className="w-3.5 h-3.5" />
              <span>Speed & Compression</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="p-6 space-y-6 max-w-7xl mx-auto w-full">
        {/* Architecture Spec Comparison Bar */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Teacher Spec */}
          <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-amber-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Layers className="w-4 h-4" />
                  Teacher Architecture
                </span>
                <span className="text-[10px] bg-amber-500/10 text-amber-400 border border-amber-500/30 px-1.5 py-0.5 rounded font-mono">
                  Full Model
                </span>
              </div>
              <div className="space-y-1 text-xs text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-400">Layers:</span>
                  <span className="font-mono font-medium">4 Layers</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Hidden Dim / Heads:</span>
                  <span className="font-mono font-medium">64 dim (4 heads)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Role:</span>
                  <span className="font-medium text-slate-200">Soft Logits Supervisor</span>
                </div>
              </div>
            </div>
            <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400 font-mono">
              <span>Status: Frozen (eval)</span>
              <span className="text-amber-400">grad = False</span>
            </div>
          </div>

          {/* Transfer Dynamics */}
          <div className="bg-gradient-to-b from-rose-950/20 to-slate-900/60 border border-rose-500/20 rounded-xl p-4 flex flex-col justify-center items-center text-center">
            <div className="w-10 h-10 rounded-full bg-rose-500/10 border border-rose-500/30 flex items-center justify-center text-rose-400 mb-2">
              <Shrink className="w-5 h-5" />
            </div>
            <h3 className="text-xs font-semibold text-slate-200">Logit Transfer Objective</h3>
            <p className="text-[11px] text-slate-400 mt-1 max-w-xs font-mono">
              L_distill = α · L_soft(τ) + (1-α) · L_hard
            </p>
            <div className="mt-3 inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-500/10 text-rose-300 text-[10px] font-mono border border-rose-500/20">
              <Sparkles className="w-3 h-3" />
              <span>Layer Dropping: [0, 2] retained</span>
            </div>
          </div>

          {/* Student Spec */}
          <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold text-emerald-400 uppercase tracking-wider flex items-center gap-1.5">
                  <Cpu className="w-4 h-4" />
                  Student Architecture
                </span>
                <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 px-1.5 py-0.5 rounded font-mono">
                  Pruned Sub-Net
                </span>
              </div>
              <div className="space-y-1 text-xs text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-400">Layers:</span>
                  <span className="font-mono font-medium">2 Layers (50% depth)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Hidden Dim / Heads:</span>
                  <span className="font-mono font-medium">64 dim (4 heads)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Inference Target:</span>
                  <span className="font-medium text-emerald-300">Low-Latency CPU</span>
                </div>
              </div>
            </div>
            <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] text-slate-400 font-mono">
              <span>Status: Trainable</span>
              <span className="text-emerald-400">AdamW active</span>
            </div>
          </div>
        </div>

        {/* SUB-TAB 1: DISTILLATION TRAINER */}
        {activeSubTab === 'trainer' && (
          <div className="space-y-6">
            {/* Presets & Hyperparameters Grid */}
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-slate-800 pb-3">
                <div className="flex items-center gap-2">
                  <Sliders className="w-4 h-4 text-rose-400" />
                  <h2 className="text-sm font-semibold text-slate-200">Distillation Hyperparameters</h2>
                </div>
                {/* Presets */}
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-xs text-slate-400 mr-1">Presets:</span>
                  {presets.map((p) => (
                    <button
                      key={p.id}
                      onClick={() => handleApplyPreset(p.id)}
                      className={`px-2.5 py-1 rounded text-xs transition-all ${
                        selectedPresetId === p.id
                          ? 'bg-rose-600/30 text-rose-200 border border-rose-500/50'
                          : 'bg-slate-800/60 hover:bg-slate-800 text-slate-300 border border-slate-700/50'
                      }`}
                    >
                      {p.name.split(' ')[0]}
                    </button>
                  ))}
                </div>
              </div>

              {/* Sliders */}
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
                {/* Temperature */}
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-855">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-400 font-medium">Temperature (τ)</span>
                    <span className="font-mono text-rose-400 font-semibold">{temperature.toFixed(1)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="10.0"
                    step="0.5"
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full accent-rose-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-400 mt-1">Softens output logits for dark knowledge</p>
                </div>

                {/* Alpha */}
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-850">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-400 font-medium">Soft Weight (α)</span>
                    <span className="font-mono text-rose-400 font-semibold">{alpha.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.0"
                    max="1.0"
                    step="0.05"
                    value={alpha}
                    onChange={(e) => setAlpha(parseFloat(e.target.value))}
                    className="w-full accent-rose-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-400 mt-1">
                    {alpha === 0.0 ? 'Hard labels only' : alpha === 1.0 ? 'Pure teacher logits' : `${Math.round(alpha*100)}% soft / ${Math.round((1-alpha)*100)}% hard`}
                  </p>
                </div>

                {/* Learning Rate */}
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-850">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-400 font-medium">Learning Rate</span>
                    <span className="font-mono text-rose-400 font-semibold">{learningRate}</span>
                  </div>
                  <input
                    type="range"
                    min="0.0005"
                    max="0.01"
                    step="0.0005"
                    value={learningRate}
                    onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                    className="w-full accent-rose-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-400 mt-1">AdamW optimizer step size</p>
                </div>

                {/* Training Steps */}
                <div className="bg-slate-950/60 p-3 rounded-lg border border-slate-850">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-slate-400 font-medium">Training Steps</span>
                    <span className="font-mono text-rose-400 font-semibold">{trainingSteps}</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="30"
                    step="5"
                    value={trainingSteps}
                    onChange={(e) => setTrainingSteps(parseInt(e.target.value))}
                    className="w-full accent-rose-500 cursor-pointer"
                  />
                  <p className="text-[10px] text-slate-400 mt-1">Instant CPU training budget (&lt;5s)</p>
                </div>
              </div>

              {/* Custom Training Text */}
              <div className="space-y-1.5">
                <label className="text-xs text-slate-400 font-medium flex items-center justify-between">
                  <span>Educational Training Text Prompt</span>
                  <span className="text-[10px] text-slate-400">Byte-encoded token stream</span>
                </label>
                <input
                  type="text"
                  value={customText}
                  onChange={(e) => setCustomText(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-rose-500 transition-colors font-mono"
                />
              </div>

              {/* Train Button */}
              <div className="flex items-center justify-between pt-2">
                <div className="text-xs text-slate-400 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                  <span>CPU Optimizer Ready (100% Free / Zero Cloud)</span>
                </div>
                <button
                  onClick={handleTrain}
                  disabled={isTraining}
                  className="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 disabled:bg-rose-900/50 text-white px-5 py-2 rounded-lg text-xs font-medium shadow-md shadow-rose-600/20 transition-all cursor-pointer"
                >
                  {isTraining ? (
                    <>
                      <RotateCcw className="w-3.5 h-3.5 animate-spin" />
                      <span>Optimizing Student...</span>
                    </>
                  ) : (
                    <>
                      <Play className="w-3.5 h-3.5 fill-current" />
                      <span>Execute Distillation Run</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* Training Results Dashboard */}
            {trainSummary && (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                  <span className="text-xs text-slate-400">Total Loss Delta</span>
                  <div className="flex items-baseline gap-2 mt-1">
                    <span className="text-lg font-mono font-bold text-slate-200">
                      {trainSummary.initialLoss.toFixed(3)}
                    </span>
                    <ArrowRight className="w-3 h-3 text-slate-400" />
                    <span className="text-lg font-mono font-bold text-emerald-400">
                      {trainSummary.finalLoss.toFixed(3)}
                    </span>
                  </div>
                  <span className="text-[10px] text-emerald-400/80 font-mono">
                    -$
                    {(trainSummary.initialLoss - trainSummary.finalLoss).toFixed(3)} loss reduction
                  </span>
                </div>

                <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                  <span className="text-xs text-slate-400">Top-1 Logit Agreement</span>
                  <div className="flex items-baseline gap-2 mt-1">
                    <span className="text-lg font-mono font-bold text-slate-200">
                      {(trainSummary.initialAgree * 100).toFixed(1)}%
                    </span>
                    <ArrowRight className="w-3 h-3 text-slate-400" />
                    <span className="text-lg font-mono font-bold text-rose-400">
                      {(trainSummary.finalAgree * 100).toFixed(1)}%
                    </span>
                  </div>
                  <span className="text-[10px] text-rose-400/80 font-mono">
                    +{trainSummary.improvementPct}% alignment
                  </span>
                </div>

                <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                  <span className="text-xs text-slate-400">Compression Factor</span>
                  <div className="text-lg font-mono font-bold text-amber-400 mt-1">
                    {compressionStats?.compression_ratio}x Smaller
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono">
                    {compressionStats?.param_reduction_pct}% fewer params
                  </span>
                </div>

                <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                  <span className="text-xs text-slate-400">Memory Saved</span>
                  <div className="text-lg font-mono font-bold text-cyan-400 mt-1">
                    {compressionStats?.memory_savings_mb} MB
                  </div>
                  <span className="text-[10px] text-slate-400 font-mono">
                    {compressionStats?.student_memory_mb} MB total student
                  </span>
                </div>
              </div>
            )}

            {/* Telemetry Progress Table */}
            {telemetry.length > 0 && (
              <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                    <BarChart2 className="w-4 h-4 text-rose-400" />
                    Distillation Convergence Telemetry (Step-by-Step)
                  </h3>
                  <span className="text-[11px] text-slate-400 font-mono">
                    {telemetry.length} Steps Recorded
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="text-[11px] uppercase bg-slate-950/80 text-slate-400 font-mono border-b border-slate-800">
                      <tr>
                        <th className="py-2 px-3">Step</th>
                        <th className="py-2 px-3">Total Loss</th>
                        <th className="py-2 px-3">Soft Loss (τ² · KL)</th>
                        <th className="py-2 px-3">Hard Loss (CE)</th>
                        <th className="py-2 px-3">Top-1 Agreement</th>
                        <th className="py-2 px-3">Student PPL</th>
                        <th className="py-2 px-3">Teacher PPL</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-850 font-mono text-xs">
                      {telemetry.map((t) => (
                        <tr key={t.step} className="hover:bg-slate-800/30 transition-colors">
                          <td className="py-2 px-3 font-semibold text-slate-400">#{t.step}</td>
                          <td className="py-2 px-3 text-rose-300 font-semibold">{t.total_loss}</td>
                          <td className="py-2 px-3 text-amber-300">{t.soft_loss}</td>
                          <td className="py-2 px-3 text-slate-300">{t.hard_loss}</td>
                          <td className="py-2 px-3">
                            <div className="flex items-center gap-2">
                              <div className="w-16 bg-slate-800 rounded-full h-1.5 overflow-hidden">
                                <div
                                  className="bg-rose-500 h-full rounded-full transition-all"
                                  style={{ width: `${Math.min(100, t.top1_agreement * 100)}%` }}
                                ></div>
                              </div>
                              <span className="text-slate-200 font-medium">
                                {(t.top1_agreement * 100).toFixed(1)}%
                              </span>
                            </div>
                          </td>
                          <td className="py-2 px-3 text-emerald-400">{t.student_ppl}</td>
                          <td className="py-2 px-3 text-slate-400">{t.teacher_ppl}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* SUB-TAB 2: DARK KNOWLEDGE EXPLORER */}
        {activeSubTab === 'dark_knowledge' && (
          <div className="space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 space-y-4">
              <div>
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Flame className="w-4 h-4 text-amber-400" />
                  Dark Knowledge & Temperature Softening Explorer
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  At $\tau = 1$, standard argmax masks the rich non-target distributions. Increasing $\tau$ reveals &quot;dark knowledge&quot;—the subtle semantic affinities between words that the student learns.
                </p>
              </div>

              {/* Prompt Input */}
              <div className="flex gap-2">
                <input
                  type="text"
                  value={darkPrompt}
                  onChange={(e) => setDarkPrompt(e.target.value)}
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-rose-500 font-mono"
                  placeholder="Type an input prompt..."
                />
                <button
                  onClick={handleAnalyzeDarkKnowledge}
                  disabled={isLoadingDark}
                  className="bg-rose-600 hover:bg-rose-500 disabled:bg-rose-900/50 text-white px-4 py-2 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all"
                >
                  {isLoadingDark ? <RotateCcw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                  <span>Inspect Distributions</span>
                </button>
              </div>

              {/* Temperature Selector */}
              <div className="flex items-center gap-3 pt-2">
                <span className="text-xs text-slate-400">Select Temperature:</span>
                {[1.0, 2.0, 5.0, 10.0].map((t) => (
                  <button
                    key={t}
                    onClick={() => setSelectedTemp(t)}
                    className={`px-3 py-1 rounded text-xs font-mono transition-all ${
                      Math.abs(selectedTemp - t) < 0.1
                        ? 'bg-amber-600 text-white font-bold shadow-md shadow-amber-600/20'
                        : 'bg-slate-800/80 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    τ = {t.toFixed(1)}
                  </button>
                ))}
              </div>
            </div>

            {/* Analysis Cards */}
            {currentAnalysis && (
              <div className="space-y-4">
                {/* Metric Summary Bar */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-3">
                    <span className="text-xs text-slate-400">KL Divergence</span>
                    <div className="text-base font-mono font-bold text-amber-400 mt-0.5">
                      {currentAnalysis.kl_divergence}
                    </div>
                    <span className="text-[10px] text-slate-400">Relative entropy</span>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-3">
                    <span className="text-xs text-slate-400">JS Divergence</span>
                    <div className="text-base font-mono font-bold text-cyan-400 mt-0.5">
                      {currentAnalysis.js_divergence}
                    </div>
                    <span className="text-[10px] text-slate-400">Symmetric divergence</span>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-3">
                    <span className="text-xs text-slate-400">Teacher Entropy</span>
                    <div className="text-base font-mono font-bold text-rose-400 mt-0.5">
                      {currentAnalysis.teacher_entropy}
                    </div>
                    <span className="text-[10px] text-slate-400">Distribution flatness</span>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-3">
                    <span className="text-xs text-slate-400">Student Entropy</span>
                    <div className="text-base font-mono font-bold text-emerald-400 mt-0.5">
                      {currentAnalysis.student_entropy}
                    </div>
                    <span className="text-[10px] text-slate-400">Student uncertainty</span>
                  </div>
                </div>

                {/* Dual Probability Distribution Bars */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Teacher Top Tokens */}
                  <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-amber-400 flex items-center gap-1.5">
                        <Layers className="w-4 h-4" />
                        Teacher Soft Distribution (τ = {selectedTemp})
                      </span>
                      <span className="text-[10px] text-slate-400 font-mono">Top-5 Next Tokens</span>
                    </div>

                    <div className="space-y-2">
                      {currentAnalysis.teacher_top_tokens.map((token, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-slate-200">
                              #{idx + 1} &apos;{token.token_str}&apos; (id:{token.token_id})
                            </span>
                            <span className="text-amber-400 font-semibold">
                              {(token.prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                            <div
                              className="bg-amber-500 h-full rounded-full transition-all duration-500"
                              style={{ width: `${Math.min(100, token.prob * 100)}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Student Top Tokens */}
                  <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-emerald-400 flex items-center gap-1.5">
                        <Cpu className="w-4 h-4" />
                        Student Soft Distribution (τ = {selectedTemp})
                      </span>
                      <span className="text-[10px] text-slate-400 font-mono">Top-5 Next Tokens</span>
                    </div>

                    <div className="space-y-2">
                      {currentAnalysis.student_top_tokens.map((token, idx) => (
                        <div key={idx} className="space-y-1">
                          <div className="flex justify-between text-xs font-mono">
                            <span className="text-slate-200">
                              #{idx + 1} &apos;{token.token_str}&apos; (id:{token.token_id})
                            </span>
                            <span className="text-emerald-400 font-semibold">
                              {(token.prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                            <div
                              className="bg-emerald-500 h-full rounded-full transition-all duration-500"
                              style={{ width: `${Math.min(100, token.prob * 100)}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* SUB-TAB 3: SPEED & COMPRESSION BENCHMARK */}
        {activeSubTab === 'benchmark' && (
          <div className="space-y-6">
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-5 flex items-center justify-between">
              <div>
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Gauge className="w-4 h-4 text-cyan-400" />
                  CPU Throughput & Prediction Agreement Benchmark
                </h2>
                <p className="text-xs text-slate-400 mt-1">
                  Evaluates forward-pass latency, tokens/second speedup, and exact prediction fidelity between teacher and student.
                </p>
              </div>
              <button
                onClick={handleRunBenchmark}
                disabled={isBenchmarking}
                className="bg-cyan-600 hover:bg-cyan-500 disabled:bg-cyan-900/50 text-white px-4 py-2 rounded-lg text-xs font-medium flex items-center gap-1.5 transition-all cursor-pointer"
              >
                {isBenchmarking ? <RotateCcw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                <span>Run Benchmark</span>
              </button>
            </div>

            {benchmarkData && (
              <div className="space-y-4">
                {/* Benchmark Metrics */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                    <span className="text-xs text-slate-400">Teacher Latency</span>
                    <div className="text-lg font-mono font-bold text-amber-400 mt-1">
                      {benchmarkData.benchmark.teacher_latency_ms} ms
                    </div>
                    <span className="text-[10px] text-slate-400">
                      {benchmarkData.benchmark.teacher_throughput_tok_per_sec} tok/sec
                    </span>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                    <span className="text-xs text-slate-400">Student Latency</span>
                    <div className="text-lg font-mono font-bold text-emerald-400 mt-1">
                      {benchmarkData.benchmark.student_latency_ms} ms
                    </div>
                    <span className="text-[10px] text-emerald-400">
                      {benchmarkData.benchmark.student_throughput_tok_per_sec} tok/sec
                    </span>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                    <span className="text-xs text-slate-400">CPU Speedup Factor</span>
                    <div className="text-lg font-mono font-bold text-cyan-400 mt-1">
                      {benchmarkData.benchmark.speedup_ratio}x Faster
                    </div>
                    <span className="text-[10px] text-cyan-400/80">Latency reduction</span>
                  </div>

                  <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-4">
                    <span className="text-xs text-slate-400">Top-1 Agreement</span>
                    <div className="text-lg font-mono font-bold text-rose-400 mt-1">
                      {benchmarkData.benchmark.top1_agreement_pct}%
                    </div>
                    <span className="text-[10px] text-rose-400/80">
                      Top-5: {benchmarkData.benchmark.top5_agreement_pct}%
                    </span>
                  </div>
                </div>

                {/* Sample Predictions Table */}
                <div className="bg-slate-900/70 border border-slate-800 rounded-xl p-5 space-y-3">
                  <h3 className="text-xs font-semibold text-slate-200">
                    Next-Token Predictions Side-by-Side
                  </h3>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs text-slate-300">
                      <thead className="text-[11px] uppercase bg-slate-950/80 text-slate-400 font-mono border-b border-slate-800">
                        <tr>
                          <th className="py-2 px-3">Validation Prompt</th>
                          <th className="py-2 px-3">Teacher Prediction</th>
                          <th className="py-2 px-3">Student Prediction</th>
                          <th className="py-2 px-3">Status</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-850 font-mono text-xs">
                        {benchmarkData.sample_predictions.map((item, idx) => (
                          <tr key={idx} className="hover:bg-slate-800/30">
                            <td className="py-2 px-3 text-slate-300 font-sans">{item.prompt}</td>
                            <td className="py-2 px-3 text-amber-300 font-bold">
                              {item.teacher_next_token}
                            </td>
                            <td className="py-2 px-3 text-emerald-300 font-bold">
                              {item.student_next_token}
                            </td>
                            <td className="py-2 px-3">
                              {item.tokens_match ? (
                                <span className="inline-flex items-center gap-1 text-emerald-400 font-sans">
                                  <CheckCircle2 className="w-3.5 h-3.5" />
                                  Match
                                </span>
                              ) : (
                                <span className="inline-flex items-center gap-1 text-slate-400 font-sans">
                                  <XCircle className="w-3.5 h-3.5 text-slate-400" />
                                  Diverged
                                </span>
                              )}
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
