'use client';

import React, { useState, useEffect } from 'react';
import { Swords, Play, Zap, Clock, DollarSign, Award, AlertCircle, CheckCircle2 } from 'lucide-react';
import { ModelMetadata, fetchModels, compareInArena, ArenaComparisonResponse } from '@/lib/api';

export default function ArenaView() {
  const [availableModels, setAvailableModels] = useState<ModelMetadata[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([
    'libra-llama-tied',
    'libra-mock-v1',
  ]);
  const [prompt, setPrompt] = useState('Explain how attention works in transformers in 2 clear sentences.');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(128);
  const [loading, setLoading] = useState(false);
  const [arenaResult, setArenaResult] = useState<ArenaComparisonResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      const data = await fetchModels();
      setAvailableModels(data);
      if (data.length >= 2) {
        setSelectedModels([data[0].id, data[1].id]);
      }
    }
    load();
  }, []);

  const handleToggleModel = (id: string) => {
    if (selectedModels.includes(id)) {
      if (selectedModels.length <= 1) return; // Keep at least one
      setSelectedModels(selectedModels.filter((m) => m !== id));
    } else {
      if (selectedModels.length >= 4) return; // Cap at 4 models
      setSelectedModels([...selectedModels, id]);
    }
  };

  const handleRunComparison = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim() || loading || selectedModels.length === 0) return;

    setLoading(true);
    setError(null);

    try {
      const targets = selectedModels.map((mId) => {
        const found = availableModels.find((m) => m.id === mId);
        return {
          model: mId,
          provider: found ? found.provider : undefined,
        };
      });

      const res = await compareInArena(prompt, targets, temperature, maxTokens);
      setArenaResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-y-auto p-4 sm:p-8">
      <div className="max-w-6xl mx-auto w-full space-y-6">
        {/* Header Title */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-5">
          <div>
            <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
              <Swords className="w-5 h-5 text-indigo-400" />
              <span>Model Comparison Arena</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Benchmark models side-by-side with high-precision TTFT, latency, token throughput, and pricing.
            </p>
          </div>

          <div className="flex items-center gap-2 bg-indigo-950/40 border border-indigo-800/40 px-3 py-1.5 rounded-lg text-xs text-indigo-300">
            <Zap className="w-3.5 h-3.5 text-indigo-400" />
            <span>Concurrent Evaluation Mode</span>
          </div>
        </div>

        {/* Configuration Bar */}
        <form onSubmit={handleRunComparison} className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Select Contending Models (Choose 2 to 4)
            </label>
            <div className="flex flex-wrap gap-2">
              {availableModels.map((m) => {
                const isChecked = selectedModels.includes(m.id);
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => handleToggleModel(m.id)}
                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs border transition-all ${
                      isChecked
                        ? 'bg-indigo-600/20 border-indigo-500 text-indigo-200 font-medium'
                        : 'bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700'
                    }`}
                  >
                    <span className={`w-2 h-2 rounded-full ${isChecked ? 'bg-indigo-400' : 'bg-slate-600'}`} />
                    <span>{m.name}</span>
                    <span className="text-[10px] text-slate-400 font-mono">({m.provider})</span>
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">
              Benchmark Prompt
            </label>
            <textarea
              rows={2}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Enter a prompt to evaluate across all selected models..."
              className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-xl p-3 text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all resize-none shadow-inner"
            />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-slate-800/80">
            <div className="flex items-center gap-6 text-xs text-slate-300">
              <div className="flex items-center gap-2">
                <span>Temp:</span>
                <input
                  type="number"
                  step="0.1"
                  min="0.0"
                  max="1.5"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-16 bg-slate-950 border border-slate-800 rounded px-2 py-1 text-right text-indigo-300"
                />
              </div>
              <div className="flex items-center gap-2">
                <span>Max Tokens:</span>
                <input
                  type="number"
                  step="32"
                  min="32"
                  max="512"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
                  className="w-20 bg-slate-950 border border-slate-800 rounded px-2 py-1 text-right text-indigo-300"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading || selectedModels.length === 0}
              className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-medium px-5 py-2 rounded-xl text-xs transition-all shadow-md shadow-indigo-600/20"
            >
              {loading ? (
                <>
                  <span className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  <span>Benchmarking...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>Execute Arena Run</span>
                </>
              )}
            </button>
          </div>
        </form>

        {error && (
          <div className="flex items-center gap-2 bg-rose-950/40 border border-rose-800/60 text-rose-300 px-4 py-3 rounded-xl text-xs">
            <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Results Grid */}
        {arenaResult && (
          <div className="space-y-6 animate-in fade-in duration-200">
            {/* Leaderboard Awards */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center shrink-0">
                  <Clock className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-[11px] text-slate-400">Fastest TTFT</div>
                  <div className="text-sm font-bold text-slate-100 truncate">
                    {arenaResult.rankings.fastest_ttft || 'N/A'}
                  </div>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-emerald-500/20 text-emerald-400 flex items-center justify-center shrink-0">
                  <Zap className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-[11px] text-slate-400">Highest Throughput</div>
                  <div className="text-sm font-bold text-slate-100 truncate">
                    {arenaResult.rankings.highest_throughput || 'N/A'}
                  </div>
                </div>
              </div>

              <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-cyan-500/20 text-cyan-400 flex items-center justify-center shrink-0">
                  <DollarSign className="w-5 h-5" />
                </div>
                <div>
                  <div className="text-[11px] text-slate-400">Most Economical</div>
                  <div className="text-sm font-bold text-slate-100 truncate">
                    {arenaResult.rankings.lowest_cost || 'N/A'}
                  </div>
                </div>
              </div>
            </div>

            {/* Model Outputs Side-by-Side */}
            <div className={`grid grid-cols-1 md:grid-cols-${Math.min(arenaResult.results.length, 3)} gap-4`}>
              {arenaResult.results.map((r) => (
                <div key={r.model} className="bg-slate-900/90 border border-slate-800 rounded-2xl overflow-hidden flex flex-col justify-between">
                  <div>
                    {/* Model Header */}
                    <div className="p-4 border-b border-slate-800/80 bg-slate-950/40 flex items-center justify-between">
                      <div>
                        <h4 className="font-semibold text-slate-100 text-sm truncate">{r.model}</h4>
                        <p className="text-[10px] text-slate-400 uppercase tracking-wider">{r.provider}</p>
                      </div>
                      {r.success ? (
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-rose-400" />
                      )}
                    </div>

                    {/* Output Text */}
                    <div className="p-4 text-xs text-slate-200 leading-relaxed min-h-[120px] whitespace-pre-wrap">
                      {r.success ? (
                        r.output_text || <span className="text-slate-500 italic">No output generated</span>
                      ) : (
                        <span className="text-rose-400">{r.error}</span>
                      )}
                    </div>
                  </div>

                  {/* Telemetry Footer */}
                  <div className="p-3 bg-slate-950/70 border-t border-slate-800 text-[11px] grid grid-cols-2 gap-2 text-slate-400">
                    <div>
                      <span>TTFT: </span>
                      <span className="text-slate-200 font-mono">{r.ttft_ms !== null ? `${r.ttft_ms.toFixed(1)}ms` : 'N/A'}</span>
                    </div>
                    <div>
                      <span>Latency: </span>
                      <span className="text-slate-200 font-mono">{r.total_latency_ms.toFixed(1)}ms</span>
                    </div>
                    <div>
                      <span>Speed: </span>
                      <span className="text-indigo-300 font-mono font-medium">{r.tokens_per_second.toFixed(1)} tok/s</span>
                    </div>
                    <div>
                      <span>Cost: </span>
                      <span className="text-emerald-400 font-mono font-medium">
                        {r.is_free ? '$0.00' : `$${r.cost_usd.toFixed(6)}`}
                      </span>
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
