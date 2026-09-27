'use client';

import React, { useState, useEffect } from 'react';
import {
  Plus,
  MessageSquare,
  BookOpen,
  Layers,
  Sparkles,
  Terminal,
  Trash2,
  Database,
  FileText,
  ShieldAlert,
  Activity,
  FileSearch,
  Compass,
  GitBranch,
  Shrink,
  Network,
  FastForward,
  Scale,
  CheckCheck,
  ChevronDown,
} from 'lucide-react';
import { ConversationSummary } from '@/lib/api';
import { MODULE_REGISTRY, getMaturityStyle } from '@/lib/module-status';

type WorkspaceTab =
  | 'chat'
  | 'arena'
  | 'rag'
  | 'corpus'
  | 'security'
  | 'observability'
  | 'batch'
  | 'document'
  | 'notebook'
  | 'long_context'
  | 'mcts'
  | 'distillation'
  | 'moe'
  | 'medusa'
  | 'kto'
  | 'verifiable_search'
  | 'self_rewarding'
  | 'grand_capstone';

interface SidebarProps {
  currentSessionId: string;
  activeTab?: WorkspaceTab;
  conversations?: ConversationSummary[];
  onSelectTab?: (tab: WorkspaceTab) => void;
  onSelectConversation?: (id: string) => void;
  onNewConversation?: () => void;
  onDeleteConversation?: (id: string) => void;
}

export default function Sidebar({
  currentSessionId,
  activeTab = 'chat',
  conversations = [],
  onSelectTab,
  onSelectConversation,
  onNewConversation,
  onDeleteConversation,
}: SidebarProps) {
  // Advanced Labs collapsed by default; state persisted in localStorage
  const [advancedExpanded, setAdvancedExpanded] = useState<boolean>(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('libra_sidebar_advanced_expanded');
      if (stored !== null) {
        setAdvancedExpanded(stored === 'true');
      }
    }
  }, []);

  const toggleAdvanced = () => {
    setAdvancedExpanded((prev) => {
      const next = !prev;
      if (typeof window !== 'undefined') {
        localStorage.setItem('libra_sidebar_advanced_expanded', next.toString());
      }
      return next;
    });
  };

  // Helper to render an item's status indicator
  const renderStatusBadge = (moduleId: string) => {
    const mod = MODULE_REGISTRY[moduleId];
    if (!mod) return null;
    const style = getMaturityStyle(mod.maturity);
    return (
      <span
        className={`ml-auto flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[9px] font-mono uppercase tracking-wider border ${style.bgColor} ${style.textColor} ${style.borderColor}`}
        title={`${mod.name}: ${mod.label}`}
      >
        <span className={`h-1.5 w-1.5 rounded-full ${style.dotColor}`} />
        <span>{mod.shortBadge}</span>
      </span>
    );
  };

  const coreModules: { id: WorkspaceTab; name: string; icon: React.ElementType }[] = [
    { id: 'chat', name: 'Interactive Chat', icon: MessageSquare },
    { id: 'arena', name: 'Model Arena', icon: Sparkles },
    { id: 'rag', name: 'Knowledge Base (RAG)', icon: Database },
    { id: 'notebook', name: 'Code Notebook', icon: Terminal },
  ];

  const advancedModules: { id: WorkspaceTab; name: string; icon: React.ElementType }[] = [
    { id: 'corpus', name: 'Corpus & SFT Lab', icon: FileText },
    { id: 'security', name: 'Security Lab', icon: ShieldAlert },
    { id: 'observability', name: 'Observability & Traces', icon: Activity },
    { id: 'batch', name: 'Batch Inference', icon: Layers },
    { id: 'document', name: 'Document OCR', icon: FileSearch },
    { id: 'long_context', name: 'Long Context & NIAH', icon: Compass },
    { id: 'mcts', name: 'MCTS Reasoning', icon: GitBranch },
    { id: 'distillation', name: 'Distillation Lab', icon: Shrink },
    { id: 'moe', name: 'MoE Sparse Lab', icon: Network },
    { id: 'medusa', name: 'Medusa Speculative', icon: FastForward },
    { id: 'kto', name: 'KTO Alignment', icon: Scale },
    { id: 'verifiable_search', name: 'Verifiable Search', icon: CheckCheck },
    { id: 'self_rewarding', name: 'Self-Rewarding', icon: Sparkles },
    { id: 'grand_capstone', name: 'Grand Capstone', icon: Layers },
  ];

  return (
    <aside className="libra-glass flex h-full w-64 shrink-0 flex-col justify-between border-r border-white/[0.07] select-none">
      {/* Brand Header */}
      <div className="border-b border-white/[0.07] p-4">
        <div className="flex items-center gap-2.5">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-white/10 bg-gradient-to-br from-indigo-500 to-violet-600 font-bold text-white shadow-lg shadow-indigo-500/20">
            ♎
          </div>
          <div>
            <h1 className="font-semibold tracking-tight text-slate-100 text-sm flex items-center gap-1.5">
              LIBRA
              <span className="text-[10px] bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 px-1.5 py-0.2 rounded font-mono font-normal">
                v0.1.0
              </span>
            </h1>
            <p className="text-[11px] text-slate-400">LLM Lab & Assistant</p>
          </div>
        </div>

        <button
          onClick={() => {
            if (onSelectTab) onSelectTab('chat');
            if (onNewConversation) onNewConversation();
          }}
          className="libra-interactive mt-4 w-full flex items-center justify-center gap-2 rounded-xl border border-white/[0.08] bg-white/[0.035] px-3 py-2 text-xs font-medium text-slate-200 shadow-sm group hover:border-indigo-400/20 hover:bg-indigo-500/[0.08]"
        >
          <Plus className="w-3.5 h-3.5 text-indigo-400 group-hover:scale-110 transition-transform" />
          <span>New Conversation</span>
        </button>
      </div>

      {/* Navigation Sections */}
      <div className="flex-1 overflow-y-auto px-3 py-3 text-xs scrollbar-thin space-y-4">
        {/* Tier 1: Core Modules */}
        <div>
          <div className="mb-1.5 flex items-center justify-between px-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400">
            <span className="flex items-center gap-1.5">
              <Layers className="w-3 h-3 text-indigo-400" />
              <span>Core Workspace</span>
            </span>
            <span className="text-[9px] text-slate-500 font-normal font-mono">Primary</span>
          </div>

          <div className="space-y-1">
            {coreModules.map((mod) => {
              const Icon = mod.icon;
              const isSelected = activeTab === mod.id;
              return (
                <button
                  key={mod.id}
                  onClick={() => onSelectTab && onSelectTab(mod.id)}
                  className={`w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left libra-interactive transition-all ${
                    isSelected
                      ? 'bg-indigo-500/[0.12] border border-indigo-400/30 shadow-[inset_2px_0_0_rgba(129,140,248,.9)] text-indigo-100 font-medium'
                      : 'text-slate-300 hover:text-white hover:bg-white/[0.04] border border-transparent'
                  }`}
                >
                  <Icon className={`w-4 h-4 shrink-0 ${isSelected ? 'text-indigo-400' : 'text-slate-400'}`} />
                  <span className="truncate">{mod.name}</span>
                  {renderStatusBadge(mod.id)}
                </button>
              );
            })}
          </div>
        </div>

        {/* Tier 2: Advanced Labs (Collapsible) */}
        <div className="border-t border-white/[0.06] pt-3">
          <button
            onClick={toggleAdvanced}
            className="w-full mb-1.5 flex items-center justify-between px-2 py-1 rounded-md text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-400 hover:text-slate-200 hover:bg-white/[0.03] transition-colors"
          >
            <span className="flex items-center gap-1.5">
              <Terminal className="w-3 h-3 text-violet-400" />
              <span>Advanced Labs ({advancedModules.length})</span>
            </span>
            <ChevronDown
              className={`w-3.5 h-3.5 text-slate-500 transition-transform duration-200 ${
                advancedExpanded ? 'rotate-180 text-violet-400' : ''
              }`}
            />
          </button>

          {advancedExpanded && (
            <div className="space-y-1 pl-1 pt-1 animate-in fade-in duration-200">
              {advancedModules.map((mod) => {
                const Icon = mod.icon;
                const isSelected = activeTab === mod.id;
                return (
                  <button
                    key={mod.id}
                    onClick={() => onSelectTab && onSelectTab(mod.id)}
                    className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-lg text-left libra-interactive transition-all ${
                      isSelected
                        ? 'bg-violet-500/[0.12] border border-violet-400/30 text-violet-100 font-medium'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-white/[0.035] border border-transparent'
                    }`}
                  >
                    <Icon className={`w-3.5 h-3.5 shrink-0 ${isSelected ? 'text-violet-400' : 'text-slate-500'}`} />
                    <span className="truncate text-[11px]">{mod.name}</span>
                    {renderStatusBadge(mod.id)}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Stored Conversations */}
        <div className="border-t border-white/[0.06] pt-3">
          <div className="flex items-center gap-1.5 text-slate-400 font-medium px-2 mb-2 tracking-wider uppercase text-[10px]">
            <MessageSquare className="w-3 h-3 text-indigo-400" />
            <span>Conversations</span>
          </div>

          <div className="space-y-0.5 max-h-40 overflow-y-auto scrollbar-thin">
            {conversations.length === 0 ? (
              <div className="px-3 py-2 text-slate-500 italic text-[11px]">
                No stored conversations
              </div>
            ) : (
              conversations.map((conv) => {
                const isSelected = activeTab === 'chat' && currentSessionId === conv.id;
                return (
                  <div
                    key={conv.id}
                    onClick={() => {
                      if (onSelectTab) onSelectTab('chat');
                      if (onSelectConversation) onSelectConversation(conv.id);
                    }}
                    className={`group flex items-center justify-between px-2.5 py-1.5 rounded-lg cursor-pointer libra-interactive transition-all ${
                      isSelected
                        ? 'bg-indigo-950/60 border border-indigo-700/60 text-indigo-200 font-medium'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/70 border border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate pr-1">
                      <span
                        className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                          isSelected ? 'bg-indigo-400' : 'bg-slate-600'
                        }`}
                      />
                      <span className="truncate text-xs">{conv.title}</span>
                    </div>

                    {onDeleteConversation && (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteConversation(conv.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 hover:text-rose-400 text-slate-400 transition-opacity p-0.5"
                        title="Delete conversation"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* System & Curriculum Milestones */}
        <div className="border-t border-white/[0.06] pt-3">
          <div className="flex items-center gap-1.5 text-slate-400 font-medium px-2 mb-2 tracking-wider uppercase text-[10px]">
            <CheckCheck className="w-3 h-3 text-emerald-400" />
            <span>Architecture Status</span>
          </div>
          <div className="space-y-1">
            <div className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-emerald-950/30 border border-emerald-800/30 text-emerald-300 text-left">
              <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" />
              <div className="truncate">
                <p className="font-medium truncate text-[11px]">All 50 Phases Verified</p>
                <p className="text-[10px] text-emerald-400/80">From First Principles</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="p-3 border-t border-white/[0.07] text-slate-400 text-[11px] space-y-2">
        <div className="flex items-center justify-between px-1">
          <span className="flex items-center gap-1 text-slate-400">
            <BookOpen className="w-3.5 h-3.5 text-slate-400" />
            Zero-Cost Policy
          </span>
          <span className="text-[10px] text-emerald-400 bg-emerald-950/50 border border-emerald-800/60 px-1.5 py-0.5 rounded font-mono">
            $0 / ₹0
          </span>
        </div>
        <div className="text-[10px] text-slate-500 px-1">
          CPU Engine: Intel i5 / 16GB RAM
        </div>
      </div>
    </aside>
  );
}