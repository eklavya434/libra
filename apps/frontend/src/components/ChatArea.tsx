'use client';

import React, { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Sliders, Zap, RotateCcw, AlertCircle, Copy, Check, Square, RefreshCw, Activity } from 'lucide-react';
import StatusBadge from './StatusBadge';
import ModelSelector from './ModelSelector';
import HyperparametersModal from './HyperparametersModal';
import StreamingMarkdown from './StreamingMarkdown';
import TokenSurprisalHeatmap from './TokenSurprisalHeatmap';
import {
  streamChat,
  ChatMessage,
  ChatTelemetry,
  fetchConversation,
  clearConversationMessages,
  API_BASE_URL,
} from '@/lib/api';

interface ExtendedMessage extends ChatMessage {
  id: string;
  timestamp: string;
  telemetry?: ChatTelemetry;
  error?: boolean;
}

interface ChatAreaProps {
  conversationId?: string;
  onConversationUpdated?: () => void;
  onSelectConversation?: (id: string) => void;
}

export default function ChatArea({
  conversationId,
  onConversationUpdated,
  onSelectConversation,
}: ChatAreaProps) {
  const [messages, setMessages] = useState<ExtendedMessage[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content:
        '👋 Welcome to **Libra**!\n\nThis is your personal AI Assistant and educational Large Language Model laboratory, built from first principles.\n\n### Core System Capabilities:\n- **First-Principles Engine**: Modern decoder-only transformer with RoPE, RMSNorm, SwiGLU, and weight tying.\n- **Multi-Turn Persistent Memory**: Powered by SQLite WAL with dynamic sliding-window context management.\n- **Advanced Hybrid RAG**: Okapi BM25 sparse search + Dense vector embeddings fused via Reciprocal Rank Fusion (RRF $k=60$), multi-factor re-ranking, and chunk deduplication.\n- **Real-Time Streaming UX**: Live token-by-token Markdown parsing, code syntax highlighting, one-click code copy, and abortable generation.\n\n```python\n# Example: Grounded RAG Query in Libra\nresponse = await libra.chat(\n    query="Explain Rotary Position Embeddings",\n    mode="hybrid",\n    use_rrf=True\n)\nprint(response.content)\n```\n\nAsk any question or test your knowledge base to begin!',
      // Keep the server and client initial render identical; live timestamps are
      // only created in client-side event handlers or effects below.
      timestamp: 'Ready',
    },
  ]);

  const [input, setInput] = useState('');
  // The mock provider is intentionally the initial model: it is always available,
  // costs nothing, and never sends conversation content to a cloud provider.
  const [selectedModel, setSelectedModel] = useState('libra-mock-v1');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [topP, setTopP] = useState(0.9);
  const [useRag, setUseRag] = useState(false);
  const [isParamsOpen, setIsParamsOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeAssistantId, setActiveAssistantId] = useState<string | null>(null);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [activeHeatmapId, setActiveHeatmapId] = useState<string | null>(null);
  const [heatmapLoadingId, setHeatmapLoadingId] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  const handleToggleHeatmap = async (msg: ExtendedMessage) => {
    if (activeHeatmapId === msg.id) {
      setActiveHeatmapId(null);
      return;
    }

    if (msg.telemetry?.tokenSequence) {
      setActiveHeatmapId(msg.id);
      return;
    }

    // On-demand sequence analysis via API
    setHeatmapLoadingId(msg.id);
    try {
      const res = await fetch(`${API_BASE_URL}/api/v1/telemetry/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: msg.content.slice(0, 500) }),
      });
      if (res.ok) {
        const seqData = await res.json();
        setMessages((prev) =>
          prev.map((m) =>
            m.id === msg.id
              ? {
                  ...m,
                  telemetry: {
                    ...(m.telemetry || {}),
                    tokenSequence: seqData,
                  },
                }
              : m
          )
        );
        setActiveHeatmapId(msg.id);
      }
    } catch (e) {
      console.warn('Failed to fetch token sequence telemetry:', e);
    } finally {
      setHeatmapLoadingId(null);
    }
  };

  // Load conversation messages when conversationId changes
  useEffect(() => {
    async function loadConv() {
      if (!conversationId) return;
      const conv = await fetchConversation(conversationId);
      if (conv) {
        if (conv.model) setSelectedModel(conv.model);
        if (conv.messages && conv.messages.length > 0) {
          setMessages(
            conv.messages.map((m) => ({
              id: m.id,
              role: m.role,
              content: m.content,
              timestamp: new Date(m.created_at).toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
              }),
            }))
          );
        } else {
          setMessages([
            {
              id: 'empty-session',
              role: 'assistant',
              content: `Session: **${conv.title}**\n\nAsk anything to begin this conversation.`,
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            },
          ]);
        }
      }
    }
    loadConv();
  }, [conversationId]);

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

  const executeStream = async (history: ChatMessage[], assistantId: string) => {
    setLoading(true);
    setActiveAssistantId(assistantId);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    await streamChat(history, selectedModel, {
      conversationId,
      useRag,
      temperature,
      maxTokens,
      signal: controller.signal,
      onConversationId: (newConvId) => {
        if (onSelectConversation && (!conversationId || conversationId !== newConvId)) {
          onSelectConversation(newConvId);
        }
      },
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
        setActiveAssistantId(null);
        abortControllerRef.current = null;
        if (onConversationUpdated) onConversationUpdated();
      },
      onError: (err) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantId
              ? {
                  ...msg,
                  content:
                    msg.content
                      ? `${msg.content}\n\n⚠️ Streaming Error: ${err.message}`
                      : `⚠️ Inference Error: ${err.message}\nEnsure the Libra backend is running on :8000. For offline use, select Libra Mock v1. Cloud models require a working provider connection.`,
                  error: true,
                }
              : msg
          )
        );
        setLoading(false);
        setActiveAssistantId(null);
        abortControllerRef.current = null;
      },
    });
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

    const nextMessages = [...messages, userMsg, initialAssistantMsg];
    setMessages(nextMessages);
    setInput('');

    // Filter out client-only welcome banners and prior error cards from LLM dialog history
    const historyForApi: ChatMessage[] = [
      ...messages.filter((m) => m.id !== 'welcome' && !m.error),
      userMsg,
    ].map((m) => ({
      role: m.role,
      content: m.content,
    }));

    await executeStream(historyForApi, assistantId);
  };

  const handleStopGeneration = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setLoading(false);
      setActiveAssistantId(null);
    }
  };

  const handleRegenerate = async () => {
    if (loading || messages.length < 2) return;

    // Find the last user message and remove subsequent assistant messages
    let lastUserIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserIdx = i;
        break;
      }
    }

    if (lastUserIdx === -1) return;

    const historyToKeep = messages.slice(0, lastUserIdx + 1);
    const assistantId = Date.now().toString();
    const newAssistantMsg: ExtendedMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    setMessages([...historyToKeep, newAssistantMsg]);

    const historyForApi: ChatMessage[] = historyToKeep.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    await executeStream(historyForApi, assistantId);
  };

  const handleClearHistory = async () => {
    if (conversationId) {
      await clearConversationMessages(conversationId);
      if (onConversationUpdated) onConversationUpdated();
    }
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
        <div className="flex items-center gap-2.5">
          <ModelSelector selectedModel={selectedModel} onSelectModel={setSelectedModel} />

          {/* RAG Grounding Toggle */}
          <button
            type="button"
            onClick={() => setUseRag(!useRag)}
            className={`flex items-center gap-1.5 text-xs rounded-lg px-2.5 py-1.5 transition-all border ${
              useRag
                ? 'bg-emerald-950/60 border-emerald-600/60 text-emerald-300 font-medium shadow-sm shadow-emerald-500/10'
                : 'text-slate-400 hover:text-slate-200 bg-slate-900/60 hover:bg-slate-900 border-slate-800'
            }`}
            title="Toggle Retrieval-Augmented Generation Grounding"
          >
            <span className={`w-2 h-2 rounded-full ${useRag ? 'bg-emerald-400 animate-pulse' : 'bg-slate-600'}`} />
            <span className="hidden sm:inline">RAG {useRag ? 'ON' : 'OFF'}</span>
          </button>

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
          {messages.map((msg, index) => {
            const isLatestAssistant =
              msg.role === 'assistant' &&
              index === messages.length - 1 &&
              !loading &&
              msg.id !== 'welcome';

            return (
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
                  {/* Message Content: StreamingMarkdown for Assistant, plain pre-wrap for User */}
                  {msg.role === 'user' ? (
                    <div className="whitespace-pre-wrap leading-relaxed">{msg.content}</div>
                  ) : (
                    <StreamingMarkdown
                      content={msg.content}
                      isStreaming={loading && msg.id === activeAssistantId}
                    />
                  )}

                  {/* Telemetry Badge on Assistant messages */}
                  {msg.telemetry && (
                    <div className="mt-3 pt-2.5 border-t border-slate-800/80 flex flex-wrap items-center gap-3 text-[10px] text-slate-400 select-none">
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

                  {/* Heatmap view if toggled */}
                  {activeHeatmapId === msg.id && msg.telemetry?.tokenSequence && (
                    <div className="mt-3">
                      <TokenSurprisalHeatmap sequence={msg.telemetry.tokenSequence} />
                    </div>
                  )}

                  <div className="flex items-center justify-between mt-2 pt-1 border-t border-slate-850/60 text-[10px]">
                    <div
                      className={`${
                        msg.role === 'user' ? 'text-indigo-200' : 'text-slate-500'
                      }`}
                    >
                      {msg.timestamp}
                    </div>

                    <div className="flex items-center gap-1">
                      {msg.role === 'assistant' && msg.content && !loading && (
                        <button
                          onClick={() => handleToggleHeatmap(msg)}
                          className={`flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded transition-all ${
                            activeHeatmapId === msg.id
                              ? 'bg-indigo-950/80 border border-indigo-500/50 text-indigo-300 font-medium'
                              : 'text-slate-400 hover:text-indigo-300 hover:bg-slate-800/60'
                          }`}
                          title="Inspect Token Surprisal & Probabilities"
                        >
                          {heatmapLoadingId === msg.id ? (
                            <RefreshCw className="w-3 h-3 animate-spin text-indigo-400" />
                          ) : (
                            <Activity className="w-3 h-3 text-indigo-400" />
                          )}
                          <span>{activeHeatmapId === msg.id ? 'Hide Tokens' : 'Inspect Tokens'}</span>
                        </button>
                      )}

                      {isLatestAssistant && (
                        <button
                          onClick={handleRegenerate}
                          className="opacity-0 group-hover:opacity-100 flex items-center gap-1 text-slate-400 hover:text-indigo-300 px-1.5 py-0.5 rounded hover:bg-slate-800/60 transition-all"
                          title="Regenerate response"
                        >
                          <RefreshCw className="w-3 h-3" />
                          <span>Regenerate</span>
                        </button>
                      )}

                      {msg.content && (
                        <button
                          onClick={() => handleCopy(msg.id, msg.content)}
                          className="opacity-0 group-hover:opacity-100 text-slate-400 hover:text-slate-200 p-1 rounded hover:bg-slate-800/60 transition-all"
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
                </div>

                {msg.role === 'user' && (
                  <div className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0 text-slate-300 mt-1 shadow-sm">
                    <User className="w-4 h-4" />
                  </div>
                )}
              </div>
            );
          })}

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
            disabled={loading}
            placeholder={loading ? 'Generating response...' : `Message ${selectedModel}...`}
            className="w-full bg-slate-900/90 border border-slate-800 hover:border-slate-700 focus:border-indigo-500 disabled:opacity-60 rounded-xl px-4 py-3 pr-12 text-sm text-slate-100 placeholder-slate-400 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all shadow-inner"
          />

          {loading ? (
            <button
              type="button"
              onClick={handleStopGeneration}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg bg-rose-600 hover:bg-rose-500 text-white flex items-center justify-center transition-all shadow-sm"
              title="Stop generating (Esc)"
            >
              <Square className="w-3.5 h-3.5 fill-current" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="absolute right-2 top-1/2 -translate-y-1/2 w-8 h-8 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-30 disabled:hover:bg-indigo-600 text-white flex items-center justify-center transition-all shadow-sm"
              title="Send message"
            >
              <Send className="w-4 h-4" />
            </button>
          )}
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
