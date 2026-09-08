'use client';

import React, { useState } from 'react';
import { Sparkles, Info, Activity, Flame, HelpCircle, ChevronRight, Zap } from 'lucide-react';
import { SequenceTelemetryData, TokenTelemetryData } from '@/lib/api';

interface TokenSurprisalHeatmapProps {
  sequence: SequenceTelemetryData;
  isStreaming?: boolean;
}

export default function TokenSurprisalHeatmap({ sequence, isStreaming }: TokenSurprisalHeatmapProps) {
  const [selectedToken, setSelectedToken] = useState<TokenTelemetryData | null>(null);

  // Helper to color-code based on Shannon surprisal bits
  // Low surprisal (<1.0 bit, p > 50%): Green/Emerald
  // Medium surprisal (1.0 to 3.0 bits, 12.5% <= p <= 50%): Amber/Yellow
  // High surprisal (>3.0 bits, p < 12.5%): Violet/Rose
  const getSurprisalBadgeClasses = (surprisal: number, isSelected: boolean) => {
    let baseColor = '';
    if (surprisal < 1.0) {
      baseColor = 'bg-emerald-950/60 border-emerald-600/40 text-emerald-300 hover:bg-emerald-900/60';
    } else if (surprisal <= 3.0) {
      baseColor = 'bg-amber-950/60 border-amber-600/40 text-amber-300 hover:bg-amber-900/60';
    } else {
      baseColor = 'bg-rose-950/60 border-rose-600/40 text-rose-300 hover:bg-rose-900/60';
    }

    const ring = isSelected ? 'ring-2 ring-indigo-400 ring-offset-1 ring-offset-slate-900 scale-105' : '';
    return `${baseColor} ${ring} transition-all duration-150`;
  };

  const getSurprisalCategory = (surprisal: number) => {
    if (surprisal < 1.0) return { label: 'High Confidence / Low Surprisal', color: 'text-emerald-400' };
    if (surprisal <= 3.0) return { label: 'Moderate Branching', color: 'text-amber-400' };
    return { label: 'Surprising / Creative Choice', color: 'text-rose-400' };
  };

  return (
    <div className="rounded-xl bg-slate-950/90 border border-slate-800/80 p-4 text-slate-200 text-xs shadow-inner space-y-4">
      {/* Header & Sequence Summary Cards */}
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-indigo-400" />
          <span className="font-semibold text-slate-100 text-sm">Token Telemetry & Surprisal Map</span>
          {isStreaming && (
            <span className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-full bg-indigo-950 border border-indigo-700/60 text-indigo-300 animate-pulse">
              <Zap className="w-2.5 h-2.5" /> Streaming
            </span>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-2 text-[11px] font-mono">
          <div className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300 flex items-center gap-1.5">
            <span className="text-slate-400">PPL:</span>
            <strong className="text-indigo-300">{sequence.perplexity?.toFixed(2) ?? '1.00'}</strong>
          </div>
          <div className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300 flex items-center gap-1.5">
            <span className="text-slate-400">Mean Surprisal:</span>
            <strong className="text-slate-200">{sequence.mean_surprisal_bits?.toFixed(2) ?? '0.00'} bits</strong>
          </div>
          <div className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-slate-300 flex items-center gap-1.5">
            <span className="text-slate-400">Mean Entropy:</span>
            <strong className="text-slate-200">{sequence.mean_entropy_bits?.toFixed(2) ?? '0.00'} bits</strong>
          </div>
          {sequence.tokens_per_second > 0 && (
            <div className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-emerald-300 flex items-center gap-1.5">
              <span>{sequence.tokens_per_second.toFixed(1)} tok/s</span>
            </div>
          )}
        </div>
      </div>

      {/* Surprisal Legend */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-[11px] text-slate-400 bg-slate-900/50 px-3 py-2 rounded-lg border border-slate-800/60">
        <div className="flex items-center gap-4">
          <span className="text-slate-400 font-medium">Surprisal Scale:</span>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-500/80 border border-emerald-400" />
            <span>&lt; 1.0 bit (&gt; 50% prob)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-500/80 border border-amber-400" />
            <span>1.0 – 3.0 bits (12.5% – 50%)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-rose-500/80 border border-rose-400" />
            <span>&gt; 3.0 bits (&lt; 12.5%)</span>
          </div>
        </div>
        <div className="text-slate-400 italic">Click any token badge to inspect candidate distribution</div>
      </div>

      {/* Interactive Token Badges Flow */}
      <div className="p-3 bg-slate-900/40 rounded-xl border border-slate-800/60 min-h-[60px] flex flex-wrap gap-1.5 items-center leading-relaxed">
        {sequence.tokens.map((tok, i) => {
          const isSelected = selectedToken?.index === tok.index;
          return (
            <button
              key={`${tok.index}-${tok.token_id}-${i}`}
              type="button"
              onClick={() => setSelectedToken(isSelected ? null : tok)}
              className={`px-2 py-0.5 rounded-md border text-xs font-mono inline-flex items-center gap-1 cursor-pointer select-none ${getSurprisalBadgeClasses(
                tok.surprisal_bits,
                isSelected
              )}`}
              title={`Token: "${tok.token_text}" | Prob: ${(tok.prob * 100).toFixed(1)}% | Surprisal: ${tok.surprisal_bits} bits`}
            >
              <span>{tok.token_text || '<space>'}</span>
              <span className="text-[9px] opacity-70">({tok.surprisal_bits.toFixed(1)})</span>
            </button>
          );
        })}
      </div>

      {/* Selected Token Inspector Drawer */}
      {selectedToken && (
        <div className="bg-slate-900 rounded-xl border border-indigo-500/40 p-4 space-y-3 shadow-lg animate-in fade-in duration-200">
          <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-700/50">
                Step #{selectedToken.index + 1}
              </span>
              <span className="text-sm font-mono text-slate-100 font-semibold">
                &ldquo;{selectedToken.token_text}&rdquo;
              </span>
              <span className="text-[11px] text-slate-400 font-mono">(ID: {selectedToken.token_id})</span>
            </div>

            <div className="flex items-center gap-2">
              <span className={`text-[11px] font-medium ${getSurprisalCategory(selectedToken.surprisal_bits).color}`}>
                {getSurprisalCategory(selectedToken.surprisal_bits).label}
              </span>
              <button
                type="button"
                onClick={() => setSelectedToken(null)}
                className="text-slate-400 hover:text-slate-200 text-xs px-2 py-0.5 rounded hover:bg-slate-800"
              >
                ✕ Close
              </button>
            </div>
          </div>

          {/* Metric Stats Grid */}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 font-mono text-xs">
            <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase">Probability</div>
              <div className="text-emerald-400 font-semibold text-sm">
                {(selectedToken.prob * 100).toFixed(2)}%
              </div>
            </div>

            <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase">Surprisal (I)</div>
              <div className="text-amber-400 font-semibold text-sm">
                {selectedToken.surprisal_bits.toFixed(3)} bits
              </div>
            </div>

            <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase">Entropy (H)</div>
              <div className="text-slate-200 font-semibold text-sm">
                {selectedToken.entropy_bits.toFixed(3)} bits
              </div>
            </div>

            <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase">Log-Prob (ln p)</div>
              <div className="text-slate-300 font-semibold text-sm">
                {selectedToken.logprob.toFixed(3)}
              </div>
            </div>

            <div className="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
              <div className="text-[10px] text-slate-400 uppercase">Step Latency</div>
              <div className="text-indigo-300 font-semibold text-sm">
                {selectedToken.latency_ms.toFixed(1)} ms
              </div>
            </div>
          </div>

          {/* Top-k Candidates Distribution Table / Bars */}
          {selectedToken.top_k && selectedToken.top_k.length > 0 && (
            <div className="space-y-2 pt-1">
              <div className="flex items-center justify-between text-[11px] text-slate-400">
                <span>Top Candidate Alternatives at Step #{selectedToken.index + 1}:</span>
                <span className="font-mono text-[10px]">Ranked by p(v | context)</span>
              </div>

              <div className="space-y-1.5">
                {selectedToken.top_k.map((cand, idx) => {
                  const isChosen = cand.token_id === selectedToken.token_id;
                  const pct = Math.max(2, Math.min(100, cand.prob * 100));

                  return (
                    <div
                      key={cand.token_id}
                      className={`p-1.5 rounded-md border text-xs font-mono flex items-center justify-between gap-3 ${
                        isChosen
                          ? 'bg-indigo-950/50 border-indigo-500/50 text-indigo-200'
                          : 'bg-slate-950/40 border-slate-800/60 text-slate-400'
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-[120px]">
                        <span className="text-[10px] text-slate-400 w-4">#{idx + 1}</span>
                        <span className="font-semibold text-slate-100">&ldquo;{cand.token_text}&rdquo;</span>
                        {isChosen && (
                          <span className="text-[9px] px-1.5 py-0.2 rounded bg-indigo-600/60 text-indigo-200">
                            Chosen
                          </span>
                        )}
                      </div>

                      <div className="flex-1 max-w-[240px] bg-slate-800/80 rounded-full h-2 overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            isChosen ? 'bg-indigo-400' : 'bg-slate-600'
                          }`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>

                      <div className="text-right min-w-[70px] text-[11px]">
                        <span className="font-medium text-slate-200">{(cand.prob * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
