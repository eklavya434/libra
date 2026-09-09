'use client';

import React, { useState, useEffect } from 'react';
import {
  Scale,
  ThumbsUp,
  ThumbsDown,
  Play,
  RotateCcw,
  Sparkles,
  BarChart2,
  Sliders,
  TrendingUp,
  Cpu,
  Layers,
  ArrowRight,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  Activity,
  Compass,
} from 'lucide-react';

interface PresetSample {
  prompt: string;
  completion: string;
  is_desirable: boolean;
}

interface Preset {
  id: string;
  name: string;
  description: string;
  beta: number;
  desirable_weight: number;
  undesirable_weight: number;
  samples?: PresetSample[];
  prompts?: string[];
}

interface KTOTelemetry {
  loss: number;
  desirable_loss: number;
  undesirable_loss: number;
  desirable_reward: number;
  undesirable_reward: number;
  reward_margin: number;
  kl_divergence: number;
  loss_aversion_ratio: number;
}

interface OnlineDPOTelemetry {
  loss: number;
  accuracy: number;
  chosen_implicit_reward: number;
  rejected_implicit_reward: number;
  reward_margin: number;
  prompt: string;
  chosen_text: string;
  rejected_text: string;
  chosen_oracle_score: number;
  rejected_oracle_score: number;
  mean_completion_tokens: number;
}

interface ParadigmMetrics {
  paradigm: string;
  average_oracle_reward: number;
  mean_implicit_reward: number;
  win_rate_vs_reference: number;
  mean_kl_divergence: number;
  mean_response_length: number;
}

interface BenchmarkResult {
  paradigms: ParadigmMetrics[];
  winner: string;
  sample_efficiency_notes: string;
}

export default function KTOView() {
  const [activeSubTab, setActiveSubTab] = useState<'kto' | 'online_dpo' | 'benchmark' | 'prospect'>('kto');
  const [presets, setPresets] = useState<Preset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('helpfulness_binary');

  // KTO Interactive State
  const [beta, setBeta] = useState<number>(0.1);
  const [lambdaD, setLambdaD] = useState<number>(1.0);
  const [lambdaU, setLambdaU] = useState<number>(1.33);
  const [samples, setSamples] = useState<PresetSample[]>([
    {
      prompt: 'How do I calculate gradient in PyTorch?',
      completion: 'Call loss.backward() and inspect tensor.grad.',
      is_desirable: true,
    },
    {
      prompt: 'How do I calculate gradient in PyTorch?',
      completion: 'PyTorch is a framework for deep learning. You should read the documentation online.',
      is_desirable: false,
    },
    {
      prompt: 'What does RMSNorm do?',
      completion: 'It normalizes activations by root mean square without subtracting mean.',
      is_desirable: true,
    },
    {
      prompt: 'What does RMSNorm do?',
      completion: 'I do not know the exact answer to your question.',
      is_desirable: false,
    },
  ]);

  const [ktoLoading, setKtoLoading] = useState(false);
  const [ktoTelemetry, setKtoTelemetry] = useState<KTOTelemetry | null>(null);

  // Online DPO Interactive State
  const [onlinePrompts, setOnlinePrompts] = useState<string>('What is attention in transformers?\nWrite a Python function to compute dot product.');
  const [onlineTemperature, setOnlineTemperature] = useState<number>(0.8);
  const [onlineLoading, setOnlineLoading] = useState(false);
  const [onlineTelemetry, setOnlineTelemetry] = useState<OnlineDPOTelemetry[]>([]);

  // Benchmark State
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResult | null>(null);

  useEffect(() => {
    fetch('/api/v1/kto/presets')
      .then((res) => res.json())
      .then((data) => {
        if (data.presets) {
          setPresets(data.presets);
        }
      })
      .catch((err) => console.error('Failed to load KTO presets', err));
  }, []);

  const handleSelectPreset = (presetId: string) => {
    setSelectedPresetId(presetId);
    const p = presets.find((item) => item.id === presetId);
    if (!p) return;
    setBeta(p.beta);
    setLambdaD(p.desirable_weight);
    setLambdaU(p.undesirable_weight);
    if (p.samples) {
      setSamples(p.samples);
    }
    if (p.prompts) {
      setOnlinePrompts(p.prompts.join('\n'));
    }
  };

  const handleToggleDesirable = (index: number) => {
    const updated = [...samples];
    updated[index].is_desirable = !updated[index].is_desirable;
    setSamples(updated);
  };

  const handleRunKTOStep = async () => {
    setKtoLoading(true);
    try {
      const res = await fetch('/api/v1/kto/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          samples,
          beta,
          desirable_weight: lambdaD,
          undesirable_weight: lambdaU,
          lr: 1e-4,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setKtoTelemetry(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setKtoLoading(false);
    }
  };

  const handleRunOnlineDPO = async () => {
    setOnlineLoading(true);
    try {
      const promptList = onlinePrompts
        .split('\n')
        .map((p) => p.trim())
        .filter(Boolean);
      const res = await fetch('/api/v1/kto/online-dpo/step', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompts: promptList,
          beta,
          temperature: onlineTemperature,
          num_candidates: 2,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setOnlineTelemetry(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setOnlineLoading(false);
    }
  };

  const handleRunBenchmark = async () => {
    setBenchmarkLoading(true);
    try {
      const res = await fetch('/api/v1/kto/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompts: [
            'Explain quantum computing simply.',
            'Write a Python function to reverse a list.',
            'What is photosynthesis?',
          ],
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setBenchmarkResult(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setBenchmarkLoading(false);
    }
  };

  // Generate SVG points for Kahneman-Tversky Prospect Theory Value Function
  const renderProspectCurve = () => {
    const width = 460;
    const height = 240;
    const originX = width / 2;
    const originY = height / 2;

    const points: string[] = [];
    for (let x = -80; x <= 80; x += 2) {
      let v = 0;
      if (x >= 0) {
        // Concave for gains: v(x) = lambda_D * (x^0.88)
        v = lambdaD * Math.pow(x, 0.85);
      } else {
        // Convex and STEEPER for losses: v(x) = -lambda_U * (|x|^0.88)
        v = -lambdaU * Math.pow(Math.abs(x), 0.85);
      }
      const plotX = originX + x * 2.4;
      const plotY = originY - v * 1.8;
      points.push(`${plotX},${plotY}`);
    }

    return (
      <svg className="w-full h-56 bg-slate-950/80 rounded-xl border border-slate-800/80 p-2 select-none" viewBox={`0 0 ${width} ${height}`}>
        {/* Axes */}
        <line x1="20" y1={originY} x2={width - 20} y2={originY} stroke="#334155" strokeWidth="1.5" strokeDasharray="3 3" />
        <line x1={originX} y1="20" x2={originX} y2={height - 20} stroke="#334155" strokeWidth="1.5" strokeDasharray="3 3" />

        {/* Axis Labels */}
        <text x={width - 70} y={originY - 8} fill="#94a3b8" fontSize="11" fontFamily="monospace">Gain (+r)</text>
        <text x="25" y={originY - 8} fill="#94a3b8" fontSize="11" fontFamily="monospace">Loss (-r)</text>
        <text x={originX + 8} y="32" fill="#10b981" fontSize="11" fontFamily="monospace">Value V(r)</text>
        <text x={originX + 8} y={height - 24} fill="#f43f5e" fontSize="11" fontFamily="monospace">Loss -V(r)</text>

        {/* Prospect Theory Curve */}
        <polyline fill="none" stroke="#6366f1" strokeWidth="3" points={points.join(' ')} strokeLinecap="round" />

        {/* Reference Point Circle */}
        <circle cx={originX} cy={originY} r="5" fill="#f59e0b" stroke="#fff" strokeWidth="1.5" />
        <text x={originX + 8} y={originY + 16} fill="#f59e0b" fontSize="10" fontFamily="sans-serif">Ref Point z_ref</text>

        {/* Annotations */}
        <text x={width - 150} y="70" fill="#34d399" fontSize="10" fontFamily="sans-serif">Slope = lambda_D ({lambdaD.toFixed(2)})</text>
        <text x="35" y={height - 50} fill="#fb7185" fontSize="10" fontFamily="sans-serif">Loss Slope = lambda_U ({lambdaU.toFixed(2)})</text>
      </svg>
    );
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 text-slate-100 overflow-y-auto">
      {/* Top Header */}
      <div className="border-b border-slate-800 bg-slate-900/60 p-6 backdrop-blur">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-rose-600 flex items-center justify-center shadow-lg shadow-amber-500/20">
              <Scale className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-xl font-bold tracking-tight text-white">Direct Alignment & KTO Lab</h1>
                <span className="text-xs bg-amber-500/20 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded-full font-mono">
                  Phase 47
                </span>
                <span className="text-xs bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 px-2 py-0.5 rounded-full font-mono">
                  Unpaired Binary
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Kahneman-Tversky Prospect Theory Optimization & On-Policy Online DPO Exploration
              </p>
            </div>
          </div>

          {/* Sub-tab Switcher */}
          <div className="flex items-center bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
            <button
              onClick={() => setActiveSubTab('kto')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                activeSubTab === 'kto'
                  ? 'bg-amber-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <ThumbsUp className="w-3.5 h-3.5" />
              <span>KTO Binary Lab</span>
            </button>
            <button
              onClick={() => setActiveSubTab('online_dpo')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                activeSubTab === 'online_dpo'
                  ? 'bg-indigo-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Compass className="w-3.5 h-3.5" />
              <span>Online DPO</span>
            </button>
            <button
              onClick={() => setActiveSubTab('prospect')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md transition-all ${
                activeSubTab === 'prospect'
                  ? 'bg-purple-600 text-white font-medium shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <TrendingUp className="w-3.5 h-3.5" />
              <span>Prospect Theory Curve</span>
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
              <span>Paradigm Arena</span>
            </button>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="p-6 space-y-6 max-w-7xl w-full mx-auto">
        {/* Preset Selector */}
        <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-amber-400" />
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">Educational Presets</span>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {presets.map((preset) => (
              <button
                key={preset.id}
                onClick={() => handleSelectPreset(preset.id)}
                className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
                  selectedPresetId === preset.id
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/50 font-medium'
                    : 'bg-slate-800/60 text-slate-400 border-slate-700 hover:text-slate-200'
                }`}
              >
                {preset.name}
              </button>
            ))}
          </div>
        </div>

        {/* Sub-Tab 1: KTO Binary Feedback Lab */}
        {activeSubTab === 'kto' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Col: Hyperparameters & Theory */}
            <div className="space-y-6 lg:col-span-1">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center gap-2 text-sm font-semibold text-white">
                  <Sliders className="w-4 h-4 text-amber-400" />
                  <span>KTO Hyperparameters</span>
                </div>

                <div className="space-y-3 text-xs">
                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>KL Penalty Temperature (beta)</span>
                      <span className="font-mono text-amber-300">{beta.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.01"
                      max="0.5"
                      step="0.01"
                      value={beta}
                      onChange={(e) => setBeta(parseFloat(e.target.value))}
                      className="w-full accent-amber-500"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Desirable Weight (lambda_D)</span>
                      <span className="font-mono text-emerald-300">{lambdaD.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="2.0"
                      step="0.1"
                      value={lambdaD}
                      onChange={(e) => setLambdaD(parseFloat(e.target.value))}
                      className="w-full accent-emerald-500"
                    />
                  </div>

                  <div>
                    <div className="flex justify-between text-slate-400 mb-1">
                      <span>Loss Aversion Weight (lambda_U)</span>
                      <span className="font-mono text-rose-300">{lambdaU.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="1.0"
                      max="3.0"
                      step="0.05"
                      value={lambdaU}
                      onChange={(e) => setLambdaU(parseFloat(e.target.value))}
                      className="w-full accent-rose-500"
                    />
                  </div>

                  <div className="bg-slate-950 p-3 rounded-lg border border-slate-800 font-mono text-[11px] text-slate-300">
                    <span className="text-slate-400">Loss Aversion Ratio:</span>{' '}
                    <span className="text-amber-400 font-bold">{(lambdaU / lambdaD).toFixed(2)}x</span>
                    <p className="text-[10px] text-slate-400 mt-1">
                      Negative outputs penalized {(lambdaU / lambdaD).toFixed(2)}x more strongly than equivalent gains.
                    </p>
                  </div>
                </div>

                <button
                  onClick={handleRunKTOStep}
                  disabled={ktoLoading}
                  className="w-full py-2.5 bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-white rounded-lg font-medium text-xs flex items-center justify-center gap-2 transition-all shadow-md shadow-amber-600/20"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>{ktoLoading ? 'Optimizing KTO Step...' : 'Execute KTO Training Step'}</span>
                </button>
              </div>

              {/* Telemetry Card */}
              {ktoTelemetry && (
                <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-white">
                    <Activity className="w-4 h-4 text-emerald-400" />
                    <span>Optimization Telemetry</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <div className="text-[10px] text-slate-400">Total KTO Loss</div>
                      <div className="font-mono font-bold text-white text-sm mt-0.5">{ktoTelemetry.loss}</div>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <div className="text-[10px] text-slate-400">Reward Margin</div>
                      <div className="font-mono font-bold text-emerald-400 text-sm mt-0.5">+{ktoTelemetry.reward_margin}</div>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <div className="text-[10px] text-slate-400">Desirable Reward</div>
                      <div className="font-mono text-emerald-300 mt-0.5">{ktoTelemetry.desirable_reward}</div>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800">
                      <div className="text-[10px] text-slate-400">Undesirable Reward</div>
                      <div className="font-mono text-rose-300 mt-0.5">{ktoTelemetry.undesirable_reward}</div>
                    </div>
                    <div className="bg-slate-950 p-2.5 rounded-lg border border-slate-800 col-span-2 flex justify-between items-center">
                      <span className="text-[10px] text-slate-400">Policy KL Drift</span>
                      <span className="font-mono text-indigo-300 text-xs">{ktoTelemetry.kl_divergence}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Right Col: Unpaired Binary Feedback Stream */}
            <div className="lg:col-span-2 space-y-4">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <ThumbsUp className="w-4 h-4 text-emerald-400" />
                    <h2 className="text-sm font-semibold text-white">Unpaired Feedback Samples</h2>
                  </div>
                  <span className="text-[11px] text-slate-400">
                    Click 👍 or 👎 to toggle desirable classification
                  </span>
                </div>

                <div className="space-y-3">
                  {samples.map((item, idx) => (
                    <div
                      key={idx}
                      className={`p-4 rounded-xl border transition-all ${
                        item.is_desirable
                          ? 'bg-emerald-950/20 border-emerald-500/40'
                          : 'bg-rose-950/20 border-rose-500/40'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="space-y-1.5 flex-1">
                          <div className="flex items-center gap-2 text-xs">
                            <span className="text-slate-400 font-mono text-[10px]">PROMPT:</span>
                            <span className="font-medium text-slate-200">{item.prompt}</span>
                          </div>
                          <div className="text-xs text-slate-300 bg-slate-950/60 p-2.5 rounded-lg font-mono border border-slate-800/80">
                            {item.completion}
                          </div>
                        </div>

                        <button
                          onClick={() => handleToggleDesirable(idx)}
                          className={`px-3 py-2 rounded-lg border text-xs font-semibold flex items-center gap-1.5 transition-all ${
                            item.is_desirable
                              ? 'bg-emerald-500/20 text-emerald-300 border-emerald-500/50 hover:bg-emerald-500/30'
                              : 'bg-rose-500/20 text-rose-300 border-rose-500/50 hover:bg-rose-500/30'
                          }`}
                        >
                          {item.is_desirable ? (
                            <>
                              <ThumbsUp className="w-3.5 h-3.5 text-emerald-400" />
                              <span>Desirable</span>
                            </>
                          ) : (
                            <>
                              <ThumbsDown className="w-3.5 h-3.5 text-rose-400" />
                              <span>Undesirable</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Sub-Tab 2: Online On-Policy DPO */}
        {activeSubTab === 'online_dpo' && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4 lg:col-span-1">
              <div className="flex items-center gap-2 text-sm font-semibold text-white">
                <Compass className="w-4 h-4 text-indigo-400" />
                <span>On-Policy Exploration Config</span>
              </div>
              <p className="text-xs text-slate-400">
                Online DPO samples multiple candidate completions from the current policy in real-time, eliminating off-policy distribution shift.
              </p>

              <div className="space-y-3 text-xs">
                <div>
                  <label className="block text-slate-400 mb-1">Sampling Prompts (one per line)</label>
                  <textarea
                    rows={4}
                    value={onlinePrompts}
                    onChange={(e) => setOnlinePrompts(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2.5 text-xs text-slate-200 font-mono focus:border-indigo-500 focus:outline-none"
                  />
                </div>

                <div>
                  <div className="flex justify-between text-slate-400 mb-1">
                    <span>Sampling Temperature</span>
                    <span className="font-mono text-indigo-300">{onlineTemperature.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.2"
                    max="1.5"
                    step="0.1"
                    value={onlineTemperature}
                    onChange={(e) => setOnlineTemperature(parseFloat(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                </div>

                <button
                  onClick={handleRunOnlineDPO}
                  disabled={onlineLoading}
                  className="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg font-medium text-xs flex items-center justify-center gap-2 transition-all shadow-md shadow-indigo-600/20"
                >
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>{onlineLoading ? 'Generating & Aligning...' : 'Rollout & Online DPO Step'}</span>
                </button>
              </div>
            </div>

            <div className="lg:col-span-2 space-y-4">
              <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-5 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-indigo-400" />
                    <h2 className="text-sm font-semibold text-white">Online Rollouts & Dynamic Preferences</h2>
                  </div>
                  <span className="text-[11px] text-slate-400">
                    Fresh on-policy generations ranked by Reward Model
                  </span>
                </div>

                {onlineTelemetry.length === 0 ? (
                  <div className="p-8 text-center text-slate-500 text-xs italic border border-dashed border-slate-800 rounded-xl">
                    No online rollouts executed yet. Click Rollout & Online DPO Step to explore.
                  </div>
                ) : (
                  <div className="space-y-4">
                    {onlineTelemetry.map((item, idx) => (
                      <div key={idx} className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-3">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-medium text-indigo-300">{item.prompt}</span>
                          <span className="text-slate-400 font-mono text-[10px]">Margin: +{item.reward_margin}</span>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                          <div className="p-3 bg-emerald-950/20 border border-emerald-500/30 rounded-lg space-y-1">
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] font-bold text-emerald-400 uppercase">CHOSEN ROLLOUT</span>
                              <span className="text-[10px] font-mono text-emerald-300">Reward: {item.chosen_oracle_score}</span>
                            </div>
                            <p className="font-mono text-slate-300 text-[11px]">{item.chosen_text}</p>
                          </div>

                          <div className="p-3 bg-rose-950/20 border border-rose-500/30 rounded-lg space-y-1">
                            <div className="flex items-center justify-between">
                              <span className="text-[10px] font-bold text-rose-400 uppercase">REJECTED ROLLOUT</span>
                              <span className="text-[10px] font-mono text-rose-300">Reward: {item.rejected_oracle_score}</span>
                            </div>
                            <p className="font-mono text-slate-300 text-[11px]">{item.rejected_text}</p>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Sub-Tab 3: Prospect Theory Curve */}
        {activeSubTab === 'prospect' && (
          <div className="space-y-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-semibold text-white flex items-center gap-2">
                    <TrendingUp className="w-5 h-5 text-purple-400" />
                    <span>Kahneman & Tversky Prospect Theory Value Function V(x)</span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Losses loom larger than gains: the psychological slope for losses (lambda_U) is steeper than for gains (lambda_D).
                  </p>
                </div>
              </div>

              {renderProspectCurve()}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1.5">
                  <span className="text-amber-400 font-bold">1. The Reference Point z_ref</span>
                  <p className="text-slate-400 text-[11px]">
                    Outcomes are not evaluated in absolute terms, but relative to a neutral reference point (E[r_theta]).
                  </p>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1.5">
                  <span className="text-emerald-400 font-bold">2. Diminishing Sensitivity</span>
                  <p className="text-slate-400 text-[11px]">
                    The marginal utility of additional gains decreases (concave in gains region).
                  </p>
                </div>
                <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-1.5">
                  <span className="text-rose-400 font-bold">3. Loss Aversion (lambda_U &gt; lambda_D)</span>
                  <p className="text-slate-400 text-[11px]">
                    Pain of losing is ~1.5x to 2x greater than pleasure of equivalent gain.
                  </p>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Sub-Tab 4: Paradigm Benchmark */}
        {activeSubTab === 'benchmark' && (
          <div className="space-y-6">
            <div className="bg-slate-900/50 border border-slate-800 rounded-xl p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-semibold text-white flex items-center gap-2">
                    <BarChart2 className="w-5 h-5 text-emerald-400" />
                    <span>Alignment Paradigm Arena</span>
                  </h2>
                  <p className="text-xs text-slate-400 mt-1">
                    Side-by-side benchmark comparing Base/SFT, Offline DPO, Online DPO, and KTO.
                  </p>
                </div>

                <button
                  onClick={handleRunBenchmark}
                  disabled={benchmarkLoading}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold flex items-center gap-1.5 shadow-md shadow-emerald-600/20"
                >
                  <Play className="w-3.5 h-3.5" />
                  <span>{benchmarkLoading ? 'Benchmarking...' : 'Run Comparative Arena'}</span>
                </button>
              </div>

              {benchmarkResult ? (
                <div className="space-y-4">
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs text-left border-collapse">
                      <thead>
                        <tr className="border-b border-slate-800 text-slate-400 font-mono">
                          <th className="py-2.5 px-3">PARADIGM</th>
                          <th className="py-2.5 px-3">ORACLE REWARD</th>
                          <th className="py-2.5 px-3">WIN RATE VS REF</th>
                          <th className="py-2.5 px-3">KL DIVERGENCE</th>
                          <th className="py-2.5 px-3">AVG TOKENS</th>
                        </tr>
                      </thead>
                      <tbody>
                        {benchmarkResult.paradigms.map((p, idx) => (
                          <tr key={idx} className="border-b border-slate-850 hover:bg-slate-900/40">
                            <td className="py-3 px-3 font-semibold text-white flex items-center gap-2">
                              {p.paradigm === benchmarkResult.winner && (
                                <span className="text-[10px] bg-amber-500/20 text-amber-300 border border-amber-500/30 px-1.5 py-0.2 rounded font-mono">
                                  WINNER
                                </span>
                              )}
                              <span>{p.paradigm}</span>
                            </td>
                            <td className="py-3 px-3 font-mono text-emerald-400">{p.average_oracle_reward}</td>
                            <td className="py-3 px-3 font-mono text-indigo-300">{(p.win_rate_vs_reference * 100).toFixed(1)}%</td>
                            <td className="py-3 px-3 font-mono text-slate-400">{p.mean_kl_divergence}</td>
                            <td className="py-3 px-3 font-mono text-slate-300">{p.mean_response_length}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  <div className="p-4 bg-slate-950 rounded-xl border border-slate-800 text-xs text-slate-300 space-y-1">
                    <span className="font-bold text-amber-400">Sample Efficiency Insight:</span>
                    <p className="text-slate-400 text-[11px]">{benchmarkResult.sample_efficiency_notes}</p>
                  </div>
                </div>
              ) : (
                <div className="p-8 text-center text-slate-500 text-xs italic border border-dashed border-slate-800 rounded-xl">
                  Click Run Comparative Arena to benchmark all alignment methodologies.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
