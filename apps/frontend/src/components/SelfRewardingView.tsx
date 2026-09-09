'use client';

import React, { useState, useEffect } from 'react';
import {
  Award,
  Sparkles,
  Scale,
  RefreshCw,
  Play,
  CheckCircle2,
  AlertCircle,
  Sliders,
  TrendingUp,
  Cpu,
  Layers,
  ArrowRight,
  ShieldCheck,
  Zap,
  BarChart3,
  Split,
  FileText,
  HelpCircle,
} from 'lucide-react';

interface Dimension {
  name: string;
  description: string;
  weight: number;
}

interface Rubric {
  name: string;
  description: string;
  dimensions: Dimension[];
}

interface PresetPrompt {
  title: string;
  prompt: string;
  rubric: string;
}

interface CandidateOutput {
  candidate_id: number;
  text: string;
  judge_score: number;
  normalized_score: number;
  critique: string;
}

interface IterationResult {
  iteration_id: number;
  prompt: string;
  candidates: CandidateOutput[];
  chosen_id: number;
  rejected_id: number;
  chosen_text: string;
  rejected_text: string;
  score_margin: number;
  dpo_loss: number;
  implicit_reward_margin: number;
  judge_critique: string;
}

interface PositionBiasAnalysis {
  consistent: boolean;
  forward_preference: string;
  reverse_preference: string;
  score_a_forward: number;
  score_b_forward: number;
  score_a_reverse: number;
  score_b_reverse: number;
  order_inconsistency_gap: number;
}

interface JudgeResponse {
  overall_score: number;
  normalized_score: number;
  dimension_scores: Record<string, number>;
  critique: string;
  is_pairwise: boolean;
  pairwise_winner?: string;
  position_bias_analysis?: PositionBiasAnalysis;
}

interface IterationMetric {
  iteration_id: number;
  win_rate: number;
  mean_judge_score: number;
  mean_margin: number;
  position_bias_rate: number;
  oracle_correlation: number;
}

interface BenchmarkResult {
  iterations: IterationMetric[];
  prompts_evaluated: number;
  summary: string;
}

export default function SelfRewardingView() {
  const [activeSubTab, setActiveSubTab] = useState<'flywheel' | 'judge' | 'benchmark'>('flywheel');
  const [rubrics, setRubrics] = useState<Rubric[]>([]);
  const [presetPrompts, setPresetPrompts] = useState<PresetPrompt[]>([]);

  // Flywheel state
  const [flywheelPrompt, setFlywheelPrompt] = useState(
    'Explain what multi-head self-attention computes in a transformer from first principles.'
  );
  const [iterationId, setIterationId] = useState<number>(1);
  const [numCandidates, setNumCandidates] = useState<number>(3);
  const [temperature, setTemperature] = useState<number>(0.8);
  const [beta, setBeta] = useState<number>(0.1);
  const [debiasPosition, setDebiasPosition] = useState<boolean>(true);
  const [flywheelLoading, setFlywheelLoading] = useState<boolean>(false);
  const [flywheelResult, setFlywheelResult] = useState<IterationResult | null>(null);

  // Judge state
  const [judgeInstruction, setJudgeInstruction] = useState(
    'Explain what attention computes in a transformer in simple terms.'
  );
  const [judgeResponseA, setJudgeResponseA] = useState(
    'Attention computes a dynamic weighted sum of values based on similarity queries and keys: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.'
  );
  const [judgeResponseB, setJudgeResponseB] = useState(
    'Attention is just when words look at each other in a sentence and find patterns.'
  );
  const [isPairwise, setIsPairwise] = useState<boolean>(true);
  const [selectedRubricName, setSelectedRubricName] = useState<string>('General Helpfulness');
  const [judgeLoading, setJudgeLoading] = useState<boolean>(false);
  const [judgeResult, setJudgeResult] = useState<JudgeResponse | null>(null);

  // Benchmark state
  const [benchmarkIterations, setBenchmarkIterations] = useState<number>(3);
  const [benchmarkLoading, setBenchmarkLoading] = useState<boolean>(false);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResult | null>(null);

  // Fetch presets on mount
  useEffect(() => {
    async function loadPresets() {
      try {
        const res = await fetch('/api/v1/self-rewarding/presets');
        if (res.ok) {
          const data = await res.json();
          setRubrics(data.rubrics || []);
          setPresetPrompts(data.prompts || []);
        }
      } catch (err) {
        console.error('Failed to load self-rewarding presets:', err);
      }
    }
    loadPresets();
  }, []);

  const handleRunFlywheelStep = async () => {
    setFlywheelLoading(true);
    try {
      const res = await fetch('/api/v1/self-rewarding/iterate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: flywheelPrompt,
          iteration_id: iterationId,
          num_candidates: numCandidates,
          temperature,
          beta,
          debias_position: debiasPosition,
        }),
      });
      if (res.ok) {
        const data: IterationResult = await res.json();
        setFlywheelResult(data);
      }
    } catch (err) {
      console.error('Flywheel step failed:', err);
    } finally {
      setFlywheelLoading(false);
    }
  };

  const handleRunJudge = async () => {
    setJudgeLoading(true);
    try {
      const payload: Record<string, any> = {
        instruction: judgeInstruction,
        response: judgeResponseA,
        rubric_name: selectedRubricName,
        debias_position: debiasPosition,
      };
      if (isPairwise) {
        payload.response_b = judgeResponseB;
      }
      const res = await fetch('/api/v1/self-rewarding/judge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (res.ok) {
        const data: JudgeResponse = await res.json();
        setJudgeResult(data);
      }
    } catch (err) {
      console.error('Judge execution failed:', err);
    } finally {
      setJudgeLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    setBenchmarkLoading(true);
    try {
      const res = await fetch('/api/v1/self-rewarding/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_iterations: benchmarkIterations,
        }),
      });
      if (res.ok) {
        const data: BenchmarkResult = await res.json();
        setBenchmarkResult(data);
      }
    } catch (err) {
      console.error('Benchmark failed:', err);
    } finally {
      setBenchmarkLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full overflow-y-auto bg-slate-950 text-slate-100 p-6 space-y-6">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-800 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <Award className="h-7 w-7 text-amber-400" />
            <h1 className="text-2xl font-bold tracking-tight text-white">
              Self-Rewarding Language Models
            </h1>
            <span className="text-xs font-mono bg-amber-950/80 text-amber-300 border border-amber-800 px-2 py-0.5 rounded">
              Phase 49
            </span>
          </div>
          <p className="text-sm text-slate-400 mt-1">
            Iterative DPO Alignment with LLM-as-a-Judge Self-Reward Flywheel (Yuan et al., Meta AI 2024)
          </p>
        </div>

        {/* Subtab Navigation */}
        <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1">
          <button
            onClick={() => setActiveSubTab('flywheel')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition ${
              activeSubTab === 'flywheel'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <RefreshCw className="h-3.5 w-3.5" />
            Iterative Flywheel
          </button>
          <button
            onClick={() => setActiveSubTab('judge')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition ${
              activeSubTab === 'judge'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Scale className="h-3.5 w-3.5" />
            LLM-as-a-Judge Studio
          </button>
          <button
            onClick={() => setActiveSubTab('benchmark')}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-md transition ${
              activeSubTab === 'benchmark'
                ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart3 className="h-3.5 w-3.5" />
            Progression Arena
          </button>
        </div>
      </div>

      {/* FLYWHEEL SUBTAB */}
      {activeSubTab === 'flywheel' && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Controls & Configuration */}
          <div className="lg:col-span-1 space-y-4">
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
              <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Sliders className="h-4 w-4 text-amber-400" />
                Flywheel Parameters
              </h2>

              {/* Preset Selector */}
              <div>
                <label className="text-xs text-slate-400 block mb-1">Preset Prompt</label>
                <select
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-amber-500/50"
                  onChange={(e) => {
                    const found = presetPrompts.find((p) => p.title === e.target.value);
                    if (found) setFlywheelPrompt(found.prompt);
                  }}
                >
                  {presetPrompts.map((p) => (
                    <option key={p.title} value={p.title}>
                      {p.title}
                    </option>
                  ))}
                </select>
              </div>

              {/* Training Prompt */}
              <div>
                <label className="text-xs text-slate-400 block mb-1">Instruction Prompt</label>
                <textarea
                  rows={3}
                  value={flywheelPrompt}
                  onChange={(e) => setFlywheelPrompt(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs text-slate-200 font-mono focus:outline-none focus:border-amber-500/50"
                />
              </div>

              {/* Iteration Stage */}
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Model Iteration (M_t)</span>
                  <span className="text-amber-400 font-mono font-bold">M_{iterationId}</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="5"
                  step="1"
                  value={iterationId}
                  onChange={(e) => setIterationId(Number(e.target.value))}
                  className="w-full accent-amber-500"
                />
                <div className="flex justify-between text-[10px] text-slate-500 mt-0.5">
                  <span>M_1 (First Self-Alignment)</span>
                  <span>M_5 (Advanced Flywheel)</span>
                </div>
              </div>

              {/* Rollout Candidates K */}
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>Candidate Rollouts (K)</span>
                  <span className="text-amber-400 font-mono">{numCandidates}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="5"
                  step="1"
                  value={numCandidates}
                  onChange={(e) => setNumCandidates(Number(e.target.value))}
                  className="w-full accent-amber-500"
                />
              </div>

              {/* DPO Beta */}
              <div>
                <div className="flex justify-between text-xs text-slate-400 mb-1">
                  <span>DPO Beta (KL Regularization)</span>
                  <span className="text-amber-400 font-mono">{beta}</span>
                </div>
                <input
                  type="range"
                  min="0.01"
                  max="0.5"
                  step="0.01"
                  value={beta}
                  onChange={(e) => setBeta(Number(e.target.value))}
                  className="w-full accent-amber-500"
                />
              </div>

              {/* Position Bias Toggle */}
              <div className="flex items-center justify-between pt-2 border-t border-slate-800/80">
                <div className="text-xs">
                  <span className="text-slate-300 block font-medium">Position-Bias Mitigation</span>
                  <span className="text-[11px] text-slate-500">Forward/reverse order averaging</span>
                </div>
                <input
                  type="checkbox"
                  checked={debiasPosition}
                  onChange={(e) => setDebiasPosition(e.target.checked)}
                  className="h-4 w-4 rounded accent-amber-500"
                />
              </div>

              {/* Run Button */}
              <button
                onClick={handleRunFlywheelStep}
                disabled={flywheelLoading}
                className="w-full mt-2 bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-slate-950 font-semibold text-xs py-2.5 rounded-lg flex items-center justify-center gap-2 transition disabled:opacity-50"
              >
                {flywheelLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Executing Flywheel Step...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 fill-slate-950" />
                    Execute Self-Rewarding Step
                  </>
                )}
              </button>
            </div>

            {/* Architecture Card */}
            <div className="bg-slate-900/60 border border-slate-800/80 rounded-xl p-4 text-xs text-slate-400 space-y-2">
              <span className="text-slate-200 font-semibold flex items-center gap-1.5">
                <Sparkles className="h-3.5 w-3.5 text-amber-400" />
                How the Flywheel Works
              </span>
              <p className="text-[11px] leading-relaxed">
                1. <strong>On-Policy Rollout:</strong> Model generates K candidate responses.
                <br />
                2. <strong>LLM-as-a-Judge:</strong> Same model rates each candidate on a 1-5 rubric scale with CoT critique.
                <br />
                3. <strong>Preference Pairing:</strong> Top candidate becomes <em>chosen</em>, bottom candidate becomes <em>rejected</em>.
                <br />
                4. <strong>DPO Update:</strong> Model updates weights to increase likelihood of chosen responses, improving both generation and judgment!
              </p>
            </div>
          </div>

          {/* Results Display */}
          <div className="lg:col-span-2 space-y-5">
            {flywheelResult ? (
              <div className="space-y-4">
                {/* Telemetry Metric Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3">
                    <span className="text-[11px] text-slate-500 block">Iteration</span>
                    <span className="text-lg font-mono font-bold text-amber-400">
                      M_{flywheelResult.iteration_id}
                    </span>
                  </div>
                  <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3">
                    <span className="text-[11px] text-slate-500 block">Judge Score Margin</span>
                    <span className="text-lg font-mono font-bold text-emerald-400">
                      +{flywheelResult.score_margin.toFixed(3)}
                    </span>
                  </div>
                  <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3">
                    <span className="text-[11px] text-slate-500 block">DPO Loss</span>
                    <span className="text-lg font-mono font-bold text-sky-400">
                      {flywheelResult.dpo_loss.toFixed(4)}
                    </span>
                  </div>
                  <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-3">
                    <span className="text-[11px] text-slate-500 block">Implicit Reward Margin</span>
                    <span className="text-lg font-mono font-bold text-purple-400">
                      {flywheelResult.implicit_reward_margin.toFixed(3)}
                    </span>
                  </div>
                </div>

                {/* Critique Box */}
                <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4">
                  <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5 mb-2">
                    <Award className="h-4 w-4 text-amber-400" />
                    Self-Judge Verdict & Critique
                  </span>
                  <p className="text-xs text-slate-300 font-mono bg-slate-950 p-3 rounded-lg border border-slate-800/80">
                    {flywheelResult.judge_critique}
                  </p>
                </div>

                {/* Candidates List */}
                <div className="space-y-3">
                  <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    Self-Evaluated Candidate Rollouts ({flywheelResult.candidates.length})
                  </h3>

                  {flywheelResult.candidates.map((cand) => {
                    const isChosen = cand.candidate_id === flywheelResult.chosen_id;
                    const isRejected = cand.candidate_id === flywheelResult.rejected_id;

                    return (
                      <div
                        key={cand.candidate_id}
                        className={`border rounded-xl p-4 transition ${
                          isChosen
                            ? 'bg-emerald-950/20 border-emerald-500/40 ring-1 ring-emerald-500/20'
                            : isRejected
                            ? 'bg-rose-950/20 border-rose-500/40'
                            : 'bg-slate-900/60 border-slate-800'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-slate-300">
                              Candidate #{cand.candidate_id + 1}
                            </span>
                            {isChosen && (
                              <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                                WINNER (CHOSEN)
                              </span>
                            )}
                            {isRejected && (
                              <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
                                REJECTED
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-mono font-bold text-amber-400">
                              {cand.judge_score.toFixed(1)} / 5.0
                            </span>
                            <span className="text-[10px] text-slate-500">
                              (norm: {cand.normalized_score.toFixed(2)})
                            </span>
                          </div>
                        </div>

                        <p className="text-xs font-mono text-slate-200 bg-slate-950/80 p-2.5 rounded-lg border border-slate-800/60 mb-2">
                          {cand.text}
                        </p>

                        <div className="text-[11px] text-slate-400 flex items-center gap-1.5">
                          <span className="text-slate-500">Judge Reason:</span>
                          <span>{cand.critique}</span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="h-96 border border-dashed border-slate-800 rounded-2xl flex flex-col items-center justify-center p-8 text-center bg-slate-900/20">
                <RefreshCw className="h-12 w-12 text-slate-700 mb-3" />
                <h3 className="text-base font-semibold text-slate-300">No Iteration Executed Yet</h3>
                <p className="text-xs text-slate-500 max-w-md mt-1">
                  Select an instruction prompt and click &quot;Execute Self-Rewarding Step&quot; to observe on-policy generation, self-scoring, and preference optimization.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* JUDGE STUDIO SUBTAB */}
      {activeSubTab === 'judge' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Inputs */}
          <div className="space-y-4">
            <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                  <Scale className="h-4 w-4 text-amber-400" />
                  Judge Configuration
                </h2>
                <div className="flex items-center gap-2">
                  <label className="text-xs text-slate-400">Pairwise Mode</label>
                  <input
                    type="checkbox"
                    checked={isPairwise}
                    onChange={(e) => setIsPairwise(e.target.checked)}
                    className="accent-amber-500 h-3.5 w-3.5"
                  />
                </div>
              </div>

              {/* Rubric Preset */}
              <div>
                <label className="text-xs text-slate-400 block mb-1">Evaluation Rubric</label>
                <select
                  value={selectedRubricName}
                  onChange={(e) => setSelectedRubricName(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-amber-500/50"
                >
                  {rubrics.map((r) => (
                    <option key={r.name} value={r.name}>
                      {r.name} — {r.description}
                    </option>
                  ))}
                </select>
              </div>

              {/* Instruction */}
              <div>
                <label className="text-xs text-slate-400 block mb-1">User Instruction</label>
                <textarea
                  rows={2}
                  value={judgeInstruction}
                  onChange={(e) => setJudgeInstruction(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-amber-500/50"
                />
              </div>

              {/* Response A */}
              <div>
                <label className="text-xs text-slate-400 block mb-1">
                  {isPairwise ? 'Candidate Response A' : 'Candidate Response'}
                </label>
                <textarea
                  rows={3}
                  value={judgeResponseA}
                  onChange={(e) => setJudgeResponseA(e.target.value)}
                  className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-amber-500/50"
                />
              </div>

              {/* Response B (if pairwise) */}
              {isPairwise && (
                <div>
                  <label className="text-xs text-slate-400 block mb-1">Candidate Response B</label>
                  <textarea
                    rows={3}
                    value={judgeResponseB}
                    onChange={(e) => setJudgeResponseB(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono focus:outline-none focus:border-amber-500/50"
                  />
                </div>
              )}

              {/* Position Debiasing */}
              {isPairwise && (
                <div className="flex items-center justify-between pt-1">
                  <span className="text-xs text-slate-400">Position-Bias Mitigation (order swap)</span>
                  <input
                    type="checkbox"
                    checked={debiasPosition}
                    onChange={(e) => setDebiasPosition(e.target.checked)}
                    className="accent-amber-500 h-3.5 w-3.5"
                  />
                </div>
              )}

              <button
                onClick={handleRunJudge}
                disabled={judgeLoading}
                className="w-full bg-gradient-to-r from-amber-600 to-amber-500 hover:from-amber-500 hover:to-amber-400 text-slate-950 font-semibold text-xs py-2.5 rounded-lg flex items-center justify-center gap-2 transition disabled:opacity-50"
              >
                {judgeLoading ? (
                  <>
                    <RefreshCw className="h-4 w-4 animate-spin" />
                    Judging Response...
                  </>
                ) : (
                  <>
                    <Scale className="h-4 w-4 fill-slate-950" />
                    Evaluate with LLM-as-a-Judge
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div className="space-y-4">
            {judgeResult ? (
              <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                  <span className="text-xs font-semibold text-slate-300">Judgment Verdict</span>
                  {judgeResult.is_pairwise && (
                    <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30">
                      Winner: {judgeResult.pairwise_winner}
                    </span>
                  )}
                </div>

                {/* Score Summary */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-500 block">Overall Score</span>
                    <span className="text-xl font-mono font-bold text-amber-400">
                      {judgeResult.overall_score.toFixed(1)} / 5.0
                    </span>
                  </div>
                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800">
                    <span className="text-[11px] text-slate-500 block">Normalized Rating</span>
                    <span className="text-xl font-mono font-bold text-emerald-400">
                      {judgeResult.normalized_score.toFixed(3)}
                    </span>
                  </div>
                </div>

                {/* Critique */}
                <div>
                  <span className="text-xs text-slate-400 block mb-1">Judge Chain-of-Thought Critique</span>
                  <p className="text-xs font-mono text-slate-300 bg-slate-950 p-3 rounded-lg border border-slate-800">
                    {judgeResult.critique}
                  </p>
                </div>

                {/* Position Bias Analysis */}
                {judgeResult.position_bias_analysis && (
                  <div className="bg-slate-950/80 border border-slate-800/80 rounded-lg p-3 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
                        <ShieldCheck className="h-3.5 w-3.5 text-amber-400" />
                        Position-Bias Mitigation Audit
                      </span>
                      <span
                        className={`text-[10px] font-mono px-1.5 py-0.5 rounded ${
                          judgeResult.position_bias_analysis.consistent
                            ? 'bg-emerald-500/20 text-emerald-300'
                            : 'bg-amber-500/20 text-amber-300'
                        }`}
                      >
                        {judgeResult.position_bias_analysis.consistent
                          ? 'Order-Consistent'
                          : 'Order-Sensitive'}
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-400">
                      <div className="bg-slate-900/60 p-2 rounded">
                        <span>Forward Order (A, B): </span>
                        <span className="text-slate-200">
                          Pref {judgeResult.position_bias_analysis.forward_preference}
                        </span>
                      </div>
                      <div className="bg-slate-900/60 p-2 rounded">
                        <span>Reverse Order (B, A): </span>
                        <span className="text-slate-200">
                          Pref {judgeResult.position_bias_analysis.reverse_preference}
                        </span>
                      </div>
                    </div>

                    <div className="text-[10px] text-slate-500">
                      Order Inconsistency Gap: {judgeResult.position_bias_analysis.order_inconsistency_gap.toFixed(4)}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-80 border border-dashed border-slate-800 rounded-xl flex flex-col items-center justify-center p-6 text-center">
                <Scale className="h-10 w-10 text-slate-700 mb-2" />
                <span className="text-xs text-slate-400">
                  Configure the response and click &quot;Evaluate with LLM-as-a-Judge&quot; to review rubric analysis.
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      {/* BENCHMARK SUBTAB */}
      {activeSubTab === 'benchmark' && (
        <div className="space-y-6">
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
            <div>
              <h2 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <BarChart3 className="h-4 w-4 text-amber-400" />
                Iterative Alignment Progression Arena
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                Evaluates how model win-rates, self-judge calibration, and position bias evolve from M_0 to M_t.
              </p>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <span className="text-xs text-slate-400">Iterations:</span>
                <select
                  value={benchmarkIterations}
                  onChange={(e) => setBenchmarkIterations(Number(e.target.value))}
                  className="bg-slate-950 border border-slate-800 rounded px-2.5 py-1 text-xs text-slate-200"
                >
                  <option value={2}>2 (M_0 vs M_1)</option>
                  <option value={3}>3 (M_0 &rarr; M_1 &rarr; M_2)</option>
                  <option value={4}>4 (M_0 &rarr; M_3)</option>
                </select>
              </div>

              <button
                onClick={handleRunBenchmark}
                disabled={benchmarkLoading}
                className="bg-amber-500 hover:bg-amber-400 text-slate-950 font-semibold text-xs px-4 py-2 rounded-lg flex items-center gap-1.5 transition disabled:opacity-50"
              >
                {benchmarkLoading ? (
                  <>
                    <RefreshCw className="h-3.5 w-3.5 animate-spin" />
                    Running Benchmark...
                  </>
                ) : (
                  <>
                    <Play className="h-3.5 w-3.5 fill-slate-950" />
                    Run Progression Benchmark
                  </>
                )}
              </button>
            </div>
          </div>

          {benchmarkResult && (
            <div className="space-y-5">
              {/* Summary */}
              <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 text-xs font-mono text-slate-300">
                {benchmarkResult.summary}
              </div>

              {/* Progression Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {benchmarkResult.iterations.map((iter) => (
                  <div
                    key={iter.iteration_id}
                    className="bg-slate-900/70 border border-slate-800 rounded-xl p-4 space-y-3"
                  >
                    <div className="flex items-center justify-between border-b border-slate-800/80 pb-2">
                      <span className="text-xs font-bold font-mono text-amber-400">
                        Iteration M_{iter.iteration_id}
                      </span>
                      <span className="text-[10px] text-slate-500">
                        {iter.iteration_id === 0 ? 'Base Model' : `Flywheel Step ${iter.iteration_id}`}
                      </span>
                    </div>

                    <div className="space-y-2">
                      <div>
                        <div className="flex justify-between text-xs text-slate-400 mb-0.5">
                          <span>Win Rate vs Base:</span>
                          <span className="font-mono font-bold text-emerald-400">{iter.win_rate}%</span>
                        </div>
                        <div className="w-full bg-slate-950 h-1.5 rounded-full overflow-hidden">
                          <div
                            className="bg-emerald-500 h-full rounded-full"
                            style={{ width: `${Math.min(100, iter.win_rate)}%` }}
                          />
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-xs text-slate-400 mb-0.5">
                          <span>Avg Judge Score:</span>
                          <span className="font-mono text-amber-300">{iter.mean_judge_score} / 5.0</span>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-xs text-slate-400 mb-0.5">
                          <span>Position Bias Inconsistency:</span>
                          <span className="font-mono text-sky-400">
                            {(iter.position_bias_rate * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>

                      <div>
                        <div className="flex justify-between text-xs text-slate-400 mb-0.5">
                          <span>Oracle Correlation:</span>
                          <span className="font-mono text-purple-400">
                            {iter.oracle_correlation >= 0 ? '+' : ''}
                            {iter.oracle_correlation.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
