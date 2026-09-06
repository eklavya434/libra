'use client';

import React from 'react';
import { Plus, MessageSquare, BookOpen, Layers, Sparkles, Terminal } from 'lucide-react';

interface SidebarProps {
  currentSessionId: string;
}

export default function Sidebar({ currentSessionId: _ }: SidebarProps) {
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
          className="mt-4 w-full flex items-center justify-center gap-2 bg-slate-900 hover:bg-slate-850 text-slate-200 border border-slate-750 hover:border-slate-700 py-2 px-3 rounded-lg text-xs font-medium transition-all shadow-sm group"
        >
          <Plus className="w-3.5 h-3.5 text-indigo-400 group-hover:scale-110 transition-transform" />
          <span>New Conversation</span>
        </button>
      </div>

      {/* Navigation Sections */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-6 text-xs">
        <div>
          <div className="flex items-center gap-1.5 text-slate-400 font-medium px-2 mb-2 tracking-wider uppercase text-[10px]">
            <Layers className="w-3 h-3 text-indigo-400" />
            <span>Educational Lab</span>
          </div>
          <div className="space-y-1">
            <button className="w-full flex items-center gap-2 px-2.5 py-2 rounded-md bg-indigo-950/40 border border-indigo-800/40 text-indigo-300 text-left">
              <Sparkles className="w-3.5 h-3.5 text-indigo-400 shrink-0" />
              <div className="truncate">
                <p className="font-medium truncate">Phase 0: Foundation</p>
                <p className="text-[10px] text-indigo-400/80">Active & Verified</p>
              </div>
            </button>

            <div className="w-full flex items-center gap-2 px-2.5 py-2 rounded-md text-slate-400 opacity-60 text-left">
              <Terminal className="w-3.5 h-3.5 shrink-0" />
              <div className="truncate">
                <p className="font-medium truncate">Phase 1: Tiny CPU LLM</p>
                <p className="text-[10px] text-slate-400">Next Milestone</p>
              </div>
            </div>
          </div>
        </div>

        <div>
          <div className="flex items-center gap-1.5 text-slate-400 font-medium px-2 mb-2 tracking-wider uppercase text-[10px]">
            <MessageSquare className="w-3 h-3 text-cyan-400" />
            <span>Recent Chats</span>
          </div>
          <div className="space-y-1">
            <button className="w-full flex items-center gap-2 px-2.5 py-2 rounded-md hover:bg-slate-900 text-slate-300 text-left transition-colors">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 shrink-0" />
              <span className="truncate">Welcome to Libra Lab</span>
            </button>
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