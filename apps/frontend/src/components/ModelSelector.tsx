'use client';

import React, { useState, useEffect, useMemo } from 'react';
import {
  Sparkles,
  ChevronDown,
  Check,
  Cpu,
  Cloud,
  Shield,
  AlertTriangle,
  Search,
  Terminal,
  Key,
  X,
  ExternalLink,
} from 'lucide-react';
import { ModelMetadata, fetchModels } from '@/lib/api';

interface ModelSelectorProps {
  selectedModel: string;
  onSelectModel: (modelId: string) => void;
}

export default function ModelSelector({ selectedModel, onSelectModel }: ModelSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'ready' | 'all' | 'ollama' | 'cloud'>('ready');
  const [searchQuery, setSearchQuery] = useState('');
  const [setupModalModel, setSetupModalModel] = useState<ModelMetadata | null>(null);

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
    id: selectedModel || '',
    name: selectedModel || 'No model selected',
    provider: 'cloud',
    architecture: 'Transformer',
    hardware_tier: 'cloud',
    is_local: false,
    available: true,
  };

  // Find the primary recommended available model (e.g. Gemini 2.5 Flash)
  const defaultReadyModel = useMemo(() => {
    const gemini = models.find((m) => m.id.includes('gemini') && m.available !== false);
    if (gemini) return gemini;
    return models.find((m) => m.available !== false) || null;
  }, [models]);

  // When no model is selected yet, or if current selection is unavailable, surface the first ready model
  useEffect(() => {
    if (models.length > 0) {
      const selected = models.find((m) => m.id === selectedModel);
      if (!selectedModel || (selected && selected.available === false)) {
        if (defaultReadyModel) onSelectModel(defaultReadyModel.id);
      }
    }
  }, [models, selectedModel, onSelectModel, defaultReadyModel]);

  const readyCount = useMemo(() => models.filter((m) => m.available !== false).length, [models]);
  const ollamaCount = useMemo(() => models.filter((m) => m.provider === 'ollama').length, [models]);
  const cloudCount = useMemo(() => models.filter((m) => !m.is_local).length, [models]);

  // Filter & sort models: available first, then alphabetically
  const filteredModels = useMemo(() => {
    let list = [...models];

    // Filter by tab
    if (activeTab === 'ready') {
      list = list.filter((m) => m.available !== false);
    } else if (activeTab === 'ollama') {
      list = list.filter((m) => m.provider === 'ollama' || m.is_local);
    } else if (activeTab === 'cloud') {
      list = list.filter((m) => !m.is_local);
    }

    // Filter by search query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      list = list.filter(
        (m) =>
          m.name.toLowerCase().includes(q) ||
          m.id.toLowerCase().includes(q) ||
          m.provider.toLowerCase().includes(q) ||
          m.architecture.toLowerCase().includes(q)
      );
    }

    // Sort: available models first, then by name
    return list.sort((a, b) => {
      const aAvail = a.available !== false ? 0 : 1;
      const bAvail = b.available !== false ? 0 : 1;
      if (aAvail !== bAvail) return aAvail - bAvail;
      return a.name.localeCompare(b.name);
    });
  }, [models, activeTab, searchQuery]);

  return (
    <div className="relative inline-block text-left">
      <button
        type="button"
        onClick={() => {
          setIsOpen(!isOpen);
          setSetupModalModel(null);
        }}
        className="flex items-center gap-2 bg-slate-900 hover:bg-slate-850 border border-slate-800 hover:border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-200 transition-all shadow-sm group"
        aria-label="Select Model"
      >
        <Sparkles className="w-3.5 h-3.5 text-indigo-400 group-hover:scale-110 transition-transform" />
        <span className="font-medium text-slate-100">{currentModel.name}</span>
        {currentModel.available === false && (
          <AlertTriangle className="w-3 h-3 text-amber-400" aria-label="needs setup" />
        )}
        <span className="text-[10px] text-slate-400 border-l border-slate-700 pl-2">
          {currentModel.is_local ? 'Local CPU' : 'Cloud'}
        </span>
        <ChevronDown
          className={`w-3.5 h-3.5 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-30"
            onClick={() => {
              setIsOpen(false);
              setSetupModalModel(null);
            }}
          />
          <div className="absolute left-0 mt-2 w-96 rounded-xl bg-slate-900 border border-slate-800 shadow-2xl shadow-black/80 z-40 overflow-hidden py-1">
            {/* Header */}
            <div className="px-3 pt-2.5 pb-2 border-b border-slate-800/80 bg-slate-950/60">
              <div className="flex items-center justify-between mb-2">
                <span className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">
                  Select Inference Model
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-950/80 border border-emerald-800/60 text-emerald-400 font-medium">
                  {readyCount} Ready to Chat
                </span>
              </div>

              {/* Search Bar */}
              <div className="relative mb-2">
                <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search 40+ models (Gemini, Llama, Qwen)..."
                  className="w-full bg-slate-900 border border-slate-750 focus:border-indigo-500 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
                {searchQuery && (
                  <button
                    onClick={() => setSearchQuery('')}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>

              {/* Category Filter Tabs */}
              <div className="flex items-center gap-1 text-[11px] font-medium pt-1">
                <button
                  type="button"
                  onClick={() => setActiveTab('ready')}
                  className={`px-2.5 py-1 rounded-md transition-colors ${
                    activeTab === 'ready'
                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/40'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  Ready ({readyCount})
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('all')}
                  className={`px-2.5 py-1 rounded-md transition-colors ${
                    activeTab === 'all'
                      ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/40'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  All ({models.length})
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('ollama')}
                  className={`px-2.5 py-1 rounded-md transition-colors ${
                    activeTab === 'ollama'
                      ? 'bg-amber-500/20 text-amber-300 border border-amber-500/40'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  Ollama ({ollamaCount})
                </button>
                <button
                  type="button"
                  onClick={() => setActiveTab('cloud')}
                  className={`px-2.5 py-1 rounded-md transition-colors ${
                    activeTab === 'cloud'
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  Cloud ({cloudCount})
                </button>
              </div>
            </div>

            {/* Setup Helper Banner (shown when a user clicks a model needing setup) */}
            {setupModalModel && (
              <div className="mx-2 my-2 p-3 rounded-lg bg-amber-950/40 border border-amber-800/60 text-xs animate-in fade-in">
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <div className="flex items-center gap-1.5 font-medium text-amber-300">
                    {setupModalModel.provider === 'ollama' ? (
                      <Terminal className="w-3.5 h-3.5" />
                    ) : (
                      <Key className="w-3.5 h-3.5" />
                    )}
                    <span>{setupModalModel.name} requires setup</span>
                  </div>
                  <button
                    onClick={() => setSetupModalModel(null)}
                    className="text-slate-400 hover:text-slate-200"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                </div>

                <p className="text-[11px] text-slate-300 mb-2 leading-relaxed">
                  {setupModalModel.provider === 'ollama' ? (
                    <>
                      This is an open-weights model designed for local CPU inference. To use it,
                      run Libra on your machine with Ollama:
                      <code className="block mt-1 p-1.5 rounded bg-slate-950 border border-slate-800 text-amber-200 font-mono text-[10px]">
                        ollama pull {setupModalModel.id}
                      </code>
                    </>
                  ) : (
                    setupModalModel.unavailable_reason ||
                    `This model requires configuring an API key in the server environment.`
                  )}
                </p>

                {defaultReadyModel && (
                  <button
                    type="button"
                    onClick={() => {
                      onSelectModel(defaultReadyModel.id);
                      setIsOpen(false);
                      setSetupModalModel(null);
                    }}
                    className="w-full py-1 px-2.5 rounded bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-[11px] flex items-center justify-center gap-1.5 shadow-sm transition-colors"
                  >
                    <Check className="w-3 h-3" />
                    Chat with {defaultReadyModel.name} instead (Ready)
                  </button>
                )}
              </div>
            )}

            {/* Model List */}
            <div className="max-h-72 overflow-y-auto py-1 divide-y divide-slate-800/40">
              {filteredModels.length === 0 ? (
                <div className="py-6 text-center text-xs text-slate-500">
                  No models found matching your search.
                </div>
              ) : (
                filteredModels.map((m) => {
                  const isSelected = m.id === selectedModel;
                  const isAvailable = m.available !== false;

                  return (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => {
                        if (isAvailable) {
                          onSelectModel(m.id);
                          setIsOpen(false);
                          setSetupModalModel(null);
                        } else {
                          setSetupModalModel(m);
                        }
                      }}
                      className={`w-full flex items-start justify-between px-3 py-2 text-left text-xs transition-colors group ${
                        isSelected
                          ? 'bg-indigo-950/40 border-l-2 border-indigo-500'
                          : isAvailable
                            ? 'hover:bg-slate-800/80 cursor-pointer'
                            : 'hover:bg-slate-800/40 cursor-pointer'
                      } ${!isAvailable ? 'opacity-70' : ''}`}
                    >
                      <div className="space-y-0.5 max-w-[70%]">
                        <div className="flex items-center gap-1.5 font-medium text-slate-100">
                          <span className="truncate">{m.name}</span>
                          {isSelected && (
                            <Check className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
                          )}
                        </div>
                        <div className="flex items-center gap-2 text-[10px] text-slate-400">
                          <span className="flex items-center gap-1">
                            {m.is_local ? (
                              <Cpu className="w-3 h-3 text-emerald-400" />
                            ) : (
                              <Cloud className="w-3 h-3 text-cyan-400" />
                            )}
                            <span className="capitalize">{m.provider}</span>
                          </span>
                          <span>•</span>
                          <span className="truncate">{m.architecture || m.hardware_tier}</span>
                        </div>
                      </div>

                      <div className="flex flex-col items-end gap-1">
                        <span
                          className={`flex items-center gap-1 text-[9px] px-2 py-0.5 rounded font-medium ${
                            isAvailable
                              ? 'bg-emerald-950/80 border border-emerald-700/80 text-emerald-300'
                              : m.provider === 'ollama'
                                ? 'bg-amber-950/60 border border-amber-800/60 text-amber-400'
                                : 'bg-slate-800/80 border border-slate-700/60 text-slate-400'
                          }`}
                        >
                          {isAvailable ? (
                            <>
                              <Check className="w-2.5 h-2.5 text-emerald-400" /> Ready
                            </>
                          ) : m.provider === 'ollama' ? (
                            <>
                              <Terminal className="w-2.5 h-2.5 text-amber-400" /> Needs Ollama
                            </>
                          ) : (
                            <>
                              <Key className="w-2.5 h-2.5 text-slate-400" /> Needs Key
                            </>
                          )}
                        </span>
                        {!isAvailable && (
                          <span className="text-[9px] text-slate-500 group-hover:text-indigo-400 underline transition-colors">
                            click for setup
                          </span>
                        )}
                      </div>
                    </button>
                  );
                })
              )}
            </div>

            {/* Footer */}
            <div className="px-3 py-2 bg-slate-950/60 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-400">
              <span>{readyCount} cloud model(s) active on web</span>
              <a
                href="https://github.com/eklavya434/libra#running-local-models"
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-1 text-indigo-400 hover:text-indigo-300"
              >
                <span>Run Ollama locally</span>
                <ExternalLink className="w-2.5 h-2.5" />
              </a>
            </div>
          </div>
        </>
      )}
    </div>
  );
}