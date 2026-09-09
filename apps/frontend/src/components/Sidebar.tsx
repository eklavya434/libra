'use client';

import React from 'react';
import { Plus, MessageSquare, BookOpen, Layers, Sparkles, Terminal, Trash2, Database, FileText, ShieldAlert, Activity, FileSearch, Compass, GitBranch, Shrink, Network, FastForward, Scale } from 'lucide-react';
import { ConversationSummary } from '@/lib/api';

interface SidebarProps {
  currentSessionId: string;
  activeTab?: 'chat' | 'arena' | 'rag' | 'corpus' | 'security' | 'observability' | 'batch' | 'document' | 'notebook' | 'long_context' | 'mcts' | 'distillation' | 'moe' | 'medusa' | 'kto';
  conversations?: ConversationSummary[];
  onSelectTab?: (tab: 'chat' | 'arena' | 'rag' | 'corpus' | 'security' | 'observability' | 'batch' | 'document' | 'notebook' | 'long_context' | 'mcts' | 'distillation' | 'moe' | 'medusa' | 'kto') => void;
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
  return (
    <aside className="w-64 bg-slate-950 border-r border-slate-800/80 flex flex-col justify-between h-full select-none">
      {/* Brand Header */}
      <div className="p-4 border-b border-slate-850">
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-indigo-600 to-violet-500 flex items-center justify-center font-bold text-white shadow-md shadow-indigo-500/20">
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
          className="mt-4 w-full flex items-center justify-center gap-2 bg-slate-900 hover:bg-slate-850 text-slate-200 border border-slate-750 hover:border-slate-700 py-2 px-3 rounded-lg text-xs font-medium transition-all shadow-sm group"
        >
          <Plus className="w-3.5 h-3.5 text-indigo-400 group-hover:scale-110 transition-transform" />
          <span>New Conversation</span>
        </button>
      </div>

      {/* Navigation Sections */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6 text-xs">
        {/* Workspace Modes */}
        <div>
          <div className="flex items-center gap-1.5 text-slate-400 font-medium px-2 mb-2 tracking-wider uppercase text-[10px]">
            <Layers className="w-3 h-3 text-indigo-400" />
            <span>Workspace</span>
          </div>
          <div className="space-y-1">
            <button
              onClick={() => onSelectTab && onSelectTab('chat')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'chat'
                  ? 'bg-indigo-600/20 border border-indigo-500/40 text-indigo-200 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <MessageSquare className={`w-4 h-4 ${activeTab === 'chat' ? 'text-indigo-400' : 'text-slate-400'}`} />
              <span>Interactive Chat</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('arena')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'arena'
                  ? 'bg-indigo-600/20 border border-indigo-500/40 text-indigo-200 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Sparkles className={`w-4 h-4 ${activeTab === 'arena' ? 'text-indigo-400' : 'text-slate-400'}`} />
              <span>Model Arena</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('rag')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'rag'
                  ? 'bg-indigo-600/20 border border-indigo-500/40 text-indigo-200 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Database className={`w-4 h-4 ${activeTab === 'rag' ? 'text-indigo-400' : 'text-slate-400'}`} />
              <span>Knowledge Base (RAG)</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('corpus')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'corpus'
                  ? 'bg-indigo-600/20 border border-indigo-500/40 text-indigo-200 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <FileText className={`w-4 h-4 ${activeTab === 'corpus' ? 'text-indigo-400' : 'text-slate-400'}`} />
              <span>Corpus & SFT Lab</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('security')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'security'
                  ? 'bg-emerald-600/20 border border-emerald-500/40 text-emerald-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <ShieldAlert className={`w-4 h-4 ${activeTab === 'security' ? 'text-emerald-400' : 'text-slate-400'}`} />
              <span>Security Lab</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('observability')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'observability'
                  ? 'bg-indigo-600/20 border border-indigo-500/40 text-indigo-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Activity className={`w-4 h-4 ${activeTab === 'observability' ? 'text-indigo-400' : 'text-slate-400'}`} />
              <span>Observability & Traces</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('batch')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'batch'
                  ? 'bg-amber-600/20 border border-amber-500/40 text-amber-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Layers className={`w-4 h-4 ${activeTab === 'batch' ? 'text-amber-400' : 'text-slate-400'}`} />
              <span>Batch Inference</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('document')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'document'
                  ? 'bg-cyan-600/20 border border-cyan-500/40 text-cyan-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <FileSearch className={`w-4 h-4 ${activeTab === 'document' ? 'text-cyan-400' : 'text-slate-400'}`} />
              <span>Document OCR</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('notebook')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'notebook'
                  ? 'bg-violet-600/20 border border-violet-500/40 text-violet-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Terminal className={`w-4 h-4 ${activeTab === 'notebook' ? 'text-violet-400' : 'text-slate-400'}`} />
              <span>Code Notebook</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('long_context')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'long_context'
                  ? 'bg-teal-600/20 border border-teal-500/40 text-teal-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Compass className={`w-4 h-4 ${activeTab === 'long_context' ? 'text-teal-400' : 'text-slate-400'}`} />
              <span>Long Context & NIAH</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('mcts')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'mcts'
                  ? 'bg-purple-600/20 border border-purple-500/40 text-purple-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <GitBranch className={`w-4 h-4 ${activeTab === 'mcts' ? 'text-purple-400' : 'text-slate-400'}`} />
              <span>MCTS Reasoning</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('distillation')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'distillation'
                  ? 'bg-rose-600/20 border border-rose-500/40 text-rose-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Shrink className={`w-4 h-4 ${activeTab === 'distillation' ? 'text-rose-400' : 'text-slate-400'}`} />
              <span>Distillation Lab</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('moe')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'moe'
                  ? 'bg-purple-600/20 border border-purple-500/40 text-purple-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Network className={`w-4 h-4 ${activeTab === 'moe' ? 'text-purple-400' : 'text-slate-400'}`} />
              <span>MoE Sparse Lab</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('medusa')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'medusa'
                  ? 'bg-amber-600/20 border border-amber-500/40 text-amber-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <FastForward className={`w-4 h-4 ${activeTab === 'medusa' ? 'text-amber-400' : 'text-slate-400'}`} />
              <span>Medusa Speculative</span>
            </button>

            <button
              onClick={() => onSelectTab && onSelectTab('kto')}
              className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-all ${
                activeTab === 'kto'
                  ? 'bg-amber-600/20 border border-amber-500/40 text-amber-300 font-medium'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60'
              }`}
            >
              <Scale className={`w-4 h-4 ${activeTab === 'kto' ? 'text-amber-400' : 'text-slate-400'}`} />
              <span>KTO Alignment</span>
            </button>
          </div>
        </div>

        {/* Recent Conversations List */}
        <div>
          <div className="flex items-center justify-between px-2 mb-2">
            <span className="flex items-center gap-1.5 text-slate-400 font-medium tracking-wider uppercase text-[10px]">
              <MessageSquare className="w-3 h-3 text-cyan-400" />
              <span>Conversations</span>
            </span>
            <span className="text-[10px] text-slate-400 font-mono">
              {conversations.length}
            </span>
          </div>
          <div className="space-y-1 max-h-48 overflow-y-auto pr-1">
            {conversations.length === 0 ? (
              <div className="px-3 py-2 text-slate-400 italic text-[11px]">
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
                    className={`group flex items-center justify-between px-2.5 py-2 rounded-lg cursor-pointer transition-all ${
                      isSelected
                        ? 'bg-indigo-950/60 border border-indigo-700/60 text-indigo-200 font-medium'
                        : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/70 border border-transparent'
                    }`}
                  >
                    <div className="flex items-center gap-2 truncate pr-1">
                      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${isSelected ? 'bg-indigo-400' : 'bg-slate-600'}`} />
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

        {/* Milestones */}
        <div>
          <div className="flex items-center gap-1.5 text-slate-400 font-medium px-2 mb-2 tracking-wider uppercase text-[10px]">
            <Terminal className="w-3 h-3 text-emerald-400" />
            <span>Milestones</span>
          </div>
          <div className="space-y-1">
            <div className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-emerald-950/30 border border-emerald-800/30 text-emerald-300 text-left">
              <span className="w-2 h-2 rounded-full bg-emerald-400 shrink-0" />
              <div className="truncate">
                <p className="font-medium truncate">Phases 0 - 11</p>
                <p className="text-[10px] text-emerald-400/80">Completed</p>
              </div>
            </div>

            <div className="w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md bg-indigo-950/40 border border-indigo-800/40 text-indigo-300 text-left">
              <span className="w-2 h-2 rounded-full bg-indigo-400 shrink-0 animate-pulse" />
              <div className="truncate">
                <p className="font-medium truncate">Phase 12: Memory</p>
                <p className="text-[10px] text-indigo-400/80">SQLite & Context</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer Info */}
      <div className="p-3 border-t border-slate-850 text-slate-400 text-[11px] space-y-2">
        <div className="flex items-center justify-between px-1">
          <span className="flex items-center gap-1 text-slate-400">
            <BookOpen className="w-3.5 h-3.5 text-slate-400" />
            Docs & ADRs
          </span>
          <span className="text-[10px] text-emerald-400 bg-emerald-950/50 border border-emerald-800/60 px-1.5 py-0.5 rounded font-mono">
            $0 / ₹0 Cost
          </span>
        </div>
        <div className="text-[10px] text-slate-400 px-1">
          Target: Intel i5 CPU / 16GB RAM
        </div>
      </div>
    </aside>
  );
}