'use client';

import React, { useState, useEffect } from 'react';
import { Sparkles, ChevronDown, Check, Cpu, Cloud, Shield } from 'lucide-react';
import { ModelMetadata, fetchModels } from '@/lib/api';

interface ModelSelectorProps {
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
}

export default function ModelSelector({ selectedModel, onSelectModel }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      setLoading(true);
      const data = await fetchModels();
      setModels(data);
      setLoading(false);
    }
    load();
  }, []);

  const currentModel = models.find((m) => m.id === selectedModel) || {
    id: selectedModel,
    name: selectedModel,
    provider: 'local',
    architecture: 'Transformer',
    hardware_tier: 'cpu-light',
    is_local: true,
  };

  return (
    <div className="relative inline-block text-left">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 bg-slate-900 hover:bg-slate-850 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 transition-all shadow-sm group"
      >
        <Sparkles className="w-3.5 h-3.5 text-indigo-400 group-hover:scale-110 transition-transform" />
        <span className="font-medium text-slate-100">{currentModel.name}</span>
        <span className="text-[10px] text-slate-400 border-l border-slate-700 pl-2">
          {currentModel.is_local ? 'Local CPU' : 'Cloud'}
        </span>
        <ChevronDown className={`w-3.5 h-3.5 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-20"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute left-0 mt-2 w-72 rounded-xl bg-slate-900 border border-slate-800 shadow-xl shadow-black/60 z-30 overflow-hidden py-1 divide-y divide-slate-800/60">
            <div className="px-3 py-2 text-[11px] font-semibold text-slate-400 uppercase tracking-wider bg-slate-950/40">
              Select Inference Model
            </div>

            <div className="max-h-72 overflow-y-auto py-1">
              {models.map((m) => {
                const isSelected = m.id === selectedModel;
                return (
                  <button
                    key={m.id}
                    onClick={() => {
                      onSelectModel(m.id);
                      setIsOpen(false);
                    }}
                    className={`w-full flex items-start justify-between px-3 py-2.5 text-left text-xs transition-colors hover:bg-slate-800/80 ${
                      isSelected ? 'bg-indigo-950/40 border-l-2 border-indigo-500' : ''
                    }`}
                  >
                    <div className="space-y-0.5">
                      <div className="flex items-center gap-1.5 font-medium text-slate-100">
                        <span>{m.name}</span>
                        {isSelected && <Check className="w-3.5 h-3.5 text-indigo-400 shrink-0" />}
                      </div>
                      <div className="flex items-center gap-2 text-[10px] text-slate-400">
                        <span className="flex items-center gap-0.5">
                          {m.is_local ? <Cpu className="w-3 h-3 text-emerald-400" /> : <Cloud className="w-3 h-3 text-cyan-400" />}
                          {m.provider}
                        </span>
                        <span>•</span>
                        <span>{m.hardware_tier}</span>
                      </div>
                    </div>

                    <span
                      className={`text-[9px] px-1.5 py-0.5 rounded font-medium ${
                        m.is_local
                          ? 'bg-emerald-950/60 border border-emerald-800/60 text-emerald-400'
                          : 'bg-indigo-950/60 border border-indigo-800/60 text-indigo-400'
                      }`}
                    >
                      {m.is_local ? '$0 / Free' : 'API'}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
