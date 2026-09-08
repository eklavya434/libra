'use client';

import React, { useState } from 'react';
import { Brain, ChevronDown, ChevronUp, Sparkles } from 'lucide-react';

interface ReasoningTraceAccordionProps {
  thoughtText: string;
  isStreaming?: boolean;
  durationMs?: number;
}

export default function ReasoningTraceAccordion({
  thoughtText,
  isStreaming = false,
  durationMs,
}: ReasoningTraceAccordionProps) {
  const [isOpen, setIsOpen] = useState(false);

  if (!thoughtText || !thoughtText.trim()) {
    return null;
  }

  const durationSec = durationMs ? (durationMs / 1000).toFixed(1) : null;

  return (
    <div className="my-2.5 rounded-xl border border-indigo-900/40 bg-slate-900/50 overflow-hidden transition-all shadow-sm">
      {/* Accordion Header */}
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3.5 py-2.5 bg-indigo-950/20 hover:bg-indigo-950/30 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-indigo-400 shrink-0" />
          <span className="text-xs font-medium text-indigo-300">
            {isStreaming ? 'Thinking...' : 'Reasoning Process'}
          </span>
          {isStreaming ? (
            <span className="w-2 h-2 rounded-full bg-indigo-400 animate-ping" />
          ) : (
            durationSec && (
              <span className="text-[10px] text-slate-400 font-mono">
                ({durationSec}s)
              </span>
            )
          )}
        </div>

        <div className="flex items-center gap-1.5 text-xs text-slate-400">
          <span className="text-[11px] font-mono">{isOpen ? 'Hide' : 'Show'} trace</span>
          {isOpen ? (
            <ChevronUp className="w-3.5 h-3.5 text-slate-400" />
          ) : (
            <ChevronDown className="w-3.5 h-3.5 text-slate-400" />
          )}
        </div>
      </button>

      {/* Accordion Body */}
      {isOpen && (
        <div className="p-3.5 text-xs text-slate-300 leading-relaxed font-mono whitespace-pre-wrap bg-slate-950/60 border-t border-indigo-950/30 divide-y divide-slate-800/40">
          <div className="flex items-center gap-1.5 pb-2 text-[10px] uppercase tracking-wider text-slate-400 font-sans font-semibold">
            <Sparkles className="w-3 h-3 text-indigo-400" />
            <span>Deliberative Chain-of-Thought</span>
          </div>
          <div className="pt-2 text-slate-300 text-[11px] leading-normal">
            {thoughtText.trim()}
          </div>
        </div>
      )}
    </div>
  );
}
