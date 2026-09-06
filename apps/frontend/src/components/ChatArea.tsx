'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sliders, Zap, RotateCcw, AlertCircle, Copy, Check } from 'lucide-react';
import StatusBadge from './StatusBadge';
import ModelSelector from './ModelSelector';
import HyperparametersModal from './HyperparametersModal';
import { streamChat, ChatMessage, ChatTelemetry } from '@/lib/api';

interface ExtendedMessage extends ChatMessage {
  id: string;
  timestamp: string;
  telemetry?: ChatTelemetry;
  error?: boolean;
}

export default function ChatArea() {
  const [messages, setMessages] = useState<ExtendedMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        '👋 Welcome to **Libra**!\n\nThis is your personal AI Assistant and educational Large Language Model laboratory, built from first principles.\n\nNow equipped with:\n- **Local & Cloud Model Selection**: Switch between your custom PyTorch model (`libra-llama-tied`), Ollama, and cloud endpoints.\n- **Real-Time Token Streaming**: Server-Sent Events (SSE) word-by-word streaming on CPU.\n- **Telemetry Feedback**: High-precision TTFT, latency, token velocity (tok/s), and real-time cost tracking.',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);

  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState('libra-llama-tied');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [topP, setTopP] = useState(0.9);
  const [isParamsOpen, setIsParamsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userText = input.trim();
    const userMsg: ExtendedMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: userText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const assistantId = (Date.now() + 1).toString();
    const initialAssistantMsg: ExtendedMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages((prev) => [...prev, userMsg, initialAssistantMsg]);
    setInput('');
    setLoading(true);

    const historyForApi: ChatMessage[] = [...messages, userMsg].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    await streamChat(historyForApi, selectedModel, {
      temperature,
      maxTokens,
      onToken: (token) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId ? { ...msg, content: msg.content + token } : msg
          )
        );
      },
      onComplete: (telemetry) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId ? { ...msg, telemetry } : msg
          )
        );
        setLoading(false);
      },
      onError: (err) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content:
                    msg.content ||
                    `⚠️ Inference Error: ${err.message}\nEnsure the Libra backend is running on :8000 or select a local mock model.`,
                  error: true,
                }
              : msg
          )
        );
        setLoading(false);
      },
    });
  };

  const handleClearHistory = () => {
    setMessages([
      {
        id: 'new-session',
        role: 'assistant',
        content: 'Conversation history cleared. Ready for a new prompt!',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-hidden">
      {/* Top Header Bar */}
      <header className="h-14 border-b border-slate-800/80 px-4 sm:px-6 flex items-center justify-between bg-slate-950/70 backdrop-blur-md z-10">
        <div className="flex items-center gap-3">
          <ModelSelector selectedModel={selectedModel} onSelectModel={setSelectedModel} />

          <button
            type="button"
            onClick={() => setIsParamsOpen(true)}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 bg-slate-900/60 hover:bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 transition-colors"
            title="Sampling Hyperparameters"
          >
            <Sliders className="w-3.5 h-3.5 text-indigo-400" />
            <span className="hidden sm:inline">Params</span>
          </button>

          <button
            type="button"
            onClick={handleClearHistory}
            className="flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 bg-slate-900/60 hover:bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1.5 transition-colors"
            title="Clear Chat History"
          >
            <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
            <span className="hidden sm:inline">Clear</span>
          </button>
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
              className={`flex gap-3.5 group ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.role === 'assistant' && (
                <div className="w-7 h-7 rounded-lg bg-indigo-600/20 border border-indigo-500/30 flex items-center justify-center shrink-0 text-indigo-400 mt-1 shadow-sm">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div
                className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm shadow-sm transition-all ${
                  msg.role === 'user'
                    ? 'bg-indigo-600 text-white rounded-tr-none shadow-indigo-600/10'
                    : msg.error
                    ? 'bg-rose-950/40 border border-rose-800/60 text-rose-200 rounded-tl-none'
                    : 'bg-slate-900/90 border border-slate-800 text-slate-200 rounded-tl-none leading-relaxed shadow-black/20'
                }`}
              >
                <div className="whitespace-pre-wrap">{msg.content || (loading && msg.role === 'assistant' ? '...' : '')}</div>

                {/* Telemetry Badge on Assistant messages */}
                {msg.telemetry && (
                  <div className="mt-2.5 pt-2 border-t border-slate-800/80 flex flex-wrap items-center gap-3 text-[10px] text-slate-400">
                    <span className="flex items-center gap-1">
                      <Zap className="w-3 h-3 text-amber-400" />
                      TTFT: <strong className="text-slate-200 font-mono">{msg.telemetry.ttftMs?.toFixed(0)}ms</strong>
                    </span>
                    <span>•</span>
                    <span>
                      Latency: <strong className="text-slate-200 font-mono">{msg.telemetry.totalLatencyMs?.toFixed(0)}ms</strong>
                    </span>
                    <span>•</span>
                    <span className="text-indigo-400 font-medium font-mono">
                      {msg.telemetry.tokensPerSecond?.toFixed(1)} tok/s
                    </span>
                    <span>•</span>
                    <span className="text-emerald-400 font-mono font-medium">
                      {msg.telemetry.isFree ? '$0.00 (Local)' : `$${msg.telemetry.totalCostUsd?.toFixed(6)}`}
                    </span>
                  </div>
                )}

                <div className="flex items-center justify-between mt-1.5">
                  <div
                    className={`text-[10px] ${
                      msg.role === 'user' ? 'text-indigo-200' : 'text-slate-500'
                    }`}
                  >
                    {msg.timestamp}
                  </div>

                  {msg.content && (
                    <button
                      onClick={() => handleCopy(msg.id, msg.content)}
                      className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-slate-200 p-0.5 rounded transition-opacity"
                      title="Copy message"
                    >
                      {copiedId === msg.id ? (
                        <Check className="w-3 h-3 text-emerald-400" />
                      ) : (
                        <Copy className="w-3 h-3" />
                      )}
                    </button>
                  )}
                </div>
              </div>

              {msg.role === 'user' && (
                <div className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0 text-slate-300 mt-1 shadow-sm">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Composer Input Area */}
      <footer className="p-4 sm:px-8 border-t border-slate-850 bg-slate-950/80 backdrop-blur-md">
        <form onSubmit={handleSend} className="max-w-3xl mx-auto relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={`Message ${selectedModel}...`}
            className="w-full bg-slate-900/90 border border-slate-800 hover:border-slate-700 focus:border-indigo-500 rounded-xl px-4 py-3 pr-12 text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all shadow-inner"
          />
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 disabled:hover:bg-indigo-600 text-white flex items-center justify-center transition-all shadow-sm"
          >
            {loading ? (
              <span className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </form>

        <div className="max-w-3xl mx-auto mt-2 flex items-center justify-between text-[11px] text-slate-400 px-1">
          <div className="flex items-center gap-2">
            <span>Model: <strong className="text-slate-300">{selectedModel}</strong></span>
            <span>•</span>
            <span>Temp: <strong className="text-slate-300">{temperature.toFixed(2)}</strong></span>
          </div>
          <span className="text-emerald-400/90">
            Zero-Cost Policy Active ($0.00)
          </span>
        </div>
      </footer>

      {/* Hyperparameters Modal */}
      <HyperparametersModal
        isOpen={isParamsOpen}
        onClose={() => setIsParamsOpen(false)}
        temperature={temperature}
        setTemperature={setTemperature}
        maxTokens={maxTokens}
        setMaxTokens={setMaxTokens}
        topP={topP}
        setTopP={setTopP}
      />
    </div>
  );
}