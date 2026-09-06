'use client';

import React, { useState } from 'react';
import { Send, Bot, User, Sparkles, Terminal, BookOpen, Layers } from 'lucide-react';
import StatusBadge from './StatusBadge';

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}

export default function ChatArea() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        '👋 Welcome to **Libra**!\n\nThis is your personal AI Assistant and educational Large Language Model laboratory, built from first principles.\n\n### What we have achieved in Phase 0:\n- **Clean Modular Architecture**: FastAPI backend + Next.js frontend + decoupled packages.\n- **Zero-Cost & CPU Safe**: Calibrated specifically for your Intel Core i5-12450H (16GB RAM) with strict disk and cost guardrails.\n- **Model Provider Abstraction**: Uniform interface ready for local educational models and external providers.\n\nIn **Phase 1**, we will begin assembling our educational decoder-only Transformer from scratch in PyTorch!',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();
    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: userText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      // Simulate response or query backend
      setTimeout(() => {
        const assistantMsg: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: `♎ **Libra Mock Engine** received your prompt:\n> "${userText}"\n\n*Phase 0 status:* System foundation verified. In **Phase 1**, we will replace this mock engine with your custom-trained CPU Transformer!`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        };
        setMessages((prev) => [...prev, assistantMsg]);
        setLoading(false);
      }, 500);
    } catch {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-hidden">
      {/* Top Header Bar */}
      <header className="h-14 border-b border-slate-800/80 px-6 flex items-center justify-between bg-slate-950/70 backdrop-blur-md z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-slate-200">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span className="font-medium">libra-mock-v1</span>
            <span className="text-[10px] text-slate-400 border-l border-slate-750 pl-2">CPU-Light</span>
          </div>

          <div className="hidden sm:flex items-center gap-1.5 text-xs text-emerald-400/90 bg-emerald-950/30 border border-emerald-800/40 px-2 py-0.5 rounded">
            <span>Learning Mode: ON</span>
          </div>
        </div>

        <div>
          <StatusBadge />
        </div>
      </header>

      {/* Messages Scrollable Viewport */}
      <div className="flex-1 overflow-y-auto px-4 py-6 sm:px-8 space-y-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3.5 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="w-7 h-7 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center shrink-0 text-indigo-400 mt-1">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div
                className={`max-w-[85%] rounded-xl px-4 py-3 text-sm shadow-sm ${
                  msg.role === 'user'
                    ? 'bg-indigo-600 text-white rounded-tr-none'
                    : 'bg-slate-900/90 border border-slate-800 text-slate-200 rounded-tl-none leading-relaxed'
                }`}
              >
                <div className="whitespace-pre-wrap">{msg.content}</div>
                <div
                  className={`text-[10px] mt-1.5 ${
                    msg.role === 'user' ? 'text-indigo-200 text-right' : 'text-slate-400 text-left'
                  }`}
                >
                  {msg.timestamp}
                </div>
              </div>

              {msg.role === 'user' && (
                <div className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0 text-slate-300 mt-1">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))}

          {loading && (
            <div className="flex gap-3.5 justify-start">
              <div className="w-7 h-7 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center shrink-0 text-indigo-400 mt-1">
                <Bot className="w-4 h-4" />
              </div>
              <div className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 text-xs text-slate-400 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
                <span>Libra is thinking...</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Composer Input Area */}
      <footer className="p-4 sm:px-8 border-t border-slate-850 bg-slate-950/80 backdrop-blur-md">
        <form onSubmit={handleSend} className="max-w-3xl mx-auto relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask Libra anything, or run an educational experiment..."
            className="w-full bg-slate-900/90 border border-slate-800 hover:border-slate-700 focus:border-indigo-500 rounded-xl px-4 py-3 pr-12 text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all shadow-inner"
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 disabled:hover:bg-indigo-600 text-white flex items-center justify-center transition-all shadow-sm"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>

        <div className="max-w-3xl mx-auto mt-2 flex items-center justify-between text-[11px] text-slate-400 px-1">
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1">
              <Terminal className="w-3 h-3 text-slate-400" />
              Backend: 127.0.0.1:8000
            </span>
            <span>•</span>
            <span className="flex items-center gap-1">
              <Layers className="w-3 h-3 text-slate-400" />
              Phase 0 Active
            </span>
          </div>
          <span className="flex items-center gap-1 text-slate-400">
            <BookOpen className="w-3 h-3 text-slate-400" />
            Learning Mode
          </span>
        </div>
      </footer>
    </div>
  );
}