'use client';

import React from 'react';
import { Sliders, X, RotateCcw } from 'lucide-react';

interface HyperparametersModalProps {
  isOpen: boolean;
  onClose: () => void;
  temperature: number;
  setTemperature: (val: number) => void;
  maxTokens: number;
  setMaxTokens: (val: number) => void;
  topP: number;
  setTopP: (val: number) => void;
}

export default function HyperparametersModal({
  isOpen,
  onClose,
  temperature,
  setTemperature,
  maxTokens,
  setMaxTokens,
  topP,
  setTopP,
}: HyperparametersModalProps) {
  if (!isOpen) return null;

  const handleReset = () => {
    setTemperature(0.7);
    setMaxTokens(512);
    setTopP(0.9);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl w-full max-w-md shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-150">
        <div className="px-5 py-4 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-2 text-slate-100 font-semibold text-sm">
            <Sliders className="w-4 h-4 text-indigo-400" />
            <span>Sampling Hyperparameters</span>
          </div>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-slate-200 p-1 rounded-lg hover:bg-slate-800 transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-6 text-xs text-slate-300">
          {/* Temperature */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="font-medium text-slate-200">Temperature</label>
              <span className="font-mono text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800/40">
                {temperature.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.5"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
            <p className="text-[11px] text-slate-400">
              Higher values (0.8+) produce more creative, varied responses. Lower values (0.2) make output focused and deterministic.
            </p>
          </div>

          {/* Max Tokens */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="font-medium text-slate-200">Max Generation Tokens</label>
              <span className="font-mono text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800/40">
                {maxTokens}
              </span>
            </div>
            <input
              type="range"
              min="64"
              max="2048"
              step="64"
              value={maxTokens}
              onChange={(e) => setMaxTokens(parseInt(e.target.value, 10))}
              className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
            <p className="text-[11px] text-slate-400">
              Maximum token length the model is permitted to generate per turn.
            </p>
          </div>

          {/* Top-P */}
          <div className="space-y-2">
            <div className="flex justify-between items-center">
              <label className="font-medium text-slate-200">Top-P (Nucleus Sampling)</label>
              <span className="font-mono text-indigo-400 bg-indigo-950/60 px-2 py-0.5 rounded border border-indigo-800/40">
                {topP.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.05"
              value={topP}
              onChange={(e) => setTopP(parseFloat(e.target.value))}
              className="w-full accent-indigo-500 cursor-pointer h-1.5 bg-slate-800 rounded-lg"
            />
            <p className="text-[11px] text-slate-400">
              Filters candidate vocabulary to cumulative probability P. Top-P = 0.9 considers the top 90% likelihood mass.
            </p>
          </div>
        </div>

        <div className="px-5 py-3 bg-slate-950/60 border-t border-slate-800/80 flex items-center justify-between">
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Defaults</span>
          </button>
          <button
            onClick={onClose}
            className="bg-indigo-600 hover:bg-indigo-500 text-white px-4 py-1.5 rounded-lg text-xs font-medium transition-colors shadow-sm"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
