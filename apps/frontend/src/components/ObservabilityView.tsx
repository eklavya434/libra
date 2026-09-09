'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  GitFork,
  Clock,
  Zap,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  ChevronRight,
  Layers,
  Terminal,
  HelpCircle,
  X,
} from 'lucide-react';
import {
  fetchTraces,
  fetchTraceDetail,
  fetchObservabilityMetrics,
  clearTraces,
  TraceSummary,
  TraceDetail,
  SpanDetail,
  ObservabilityMetrics,
} from '@/lib/api';

export default function ObservabilityView() {
  const [traces, setTraces] = useState<TraceSummary[]>([]);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);
  const [activeTrace, setActiveTrace] = useState<TraceDetail | null>(null);
  const [selectedSpan, setSelectedSpan] = useState<SpanDetail | null>(null);
  const [metrics, setMetrics] = useState<ObservabilityMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'waterfall' | 'guide'>('waterfall');

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const [traceData, metricData] = await Promise.all([
        fetchTraces(30, 0),
        fetchObservabilityMetrics(),
      ]);
      setTraces(traceData.traces);
      setMetrics(metricData);

      if (traceData.traces.length > 0 && !selectedTraceId) {
        setSelectedTraceId(traceData.traces[0].trace_id);
      }
    } catch (e) {
      console.warn('Failed to load observability data:', e);
    } finally {
      setLoading(false);
    }
  }, [selectedTraceId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (!selectedTraceId) return;
    async function loadTrace() {
      setDetailLoading(true);
      try {
        const detail = await fetchTraceDetail(selectedTraceId!);
        setActiveTrace(detail);
        if (detail.spans.length > 0) {
          setSelectedSpan(detail.spans[0]);
        }
      } catch (e) {
        console.error('Failed to load trace detail:', e);
      } finally {
        setDetailLoading(false);
      }
    }
    loadTrace();
  }, [selectedTraceId]);

  const handleClear = async () => {
    try {
      await clearTraces();
      setTraces([]);
      setActiveTrace(null);
      setSelectedSpan(null);
      setSelectedTraceId(null);
      await loadData();
    } catch (e) {
      console.error('Clear error:', e);
    }
  };

  const getSpanColor = (name: string, status: string) => {
    if (status === 'ERROR') return 'bg-rose-500 border-rose-400';
    if (name.startsWith('GET') || name.startsWith('POST')) return 'bg-indigo-500 border-indigo-400';
    if (name.includes('model') || name.includes('generate') || name.includes('chat'))
      return 'bg-violet-500 border-violet-400';
    if (name.includes('rag') || name.includes('retriev') || name.includes('search'))
      return 'bg-emerald-500 border-emerald-400';
    if (name.includes('security') || name.includes('guard') || name.includes('redact'))
      return 'bg-amber-500 border-amber-400';
    return 'bg-cyan-500 border-cyan-400';
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-hidden text-slate-100">
      {/* Top Header Bar */}
      <header className="h-14 border-b border-slate-800/80 px-6 flex items-center justify-between bg-slate-950/70 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-sm font-semibold tracking-tight text-slate-100 flex items-center gap-2">
              Distributed Tracing & OpenTelemetry
              <span className="text-[10px] bg-indigo-500/20 text-indigo-400 border border-indigo-500/30 px-2 py-0.5 rounded-full font-mono">
                Phase 38
              </span>
            </h1>
            <p className="text-xs text-slate-400">Latency waterfalls, W3C traceparent propagation & token velocity</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex bg-slate-900 border border-slate-800 rounded-lg p-1 text-xs">
            <button
              onClick={() => setActiveTab('waterfall')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                activeTab === 'waterfall' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Latency Waterfall
            </button>
            <button
              onClick={() => setActiveTab('guide')}
              className={`px-3 py-1 rounded-md font-medium transition-all ${
                activeTab === 'guide' ? 'bg-indigo-600 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Tracing Guide
            </button>
          </div>

          <button
            onClick={loadData}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-900 hover:bg-slate-800 border border-slate-800 rounded-lg text-xs font-medium text-slate-300 transition-all"
          >
            <RotateCcw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>

          <button
            onClick={handleClear}
            className="px-3 py-1.5 bg-rose-950/30 hover:bg-rose-900/40 border border-rose-900/40 rounded-lg text-xs font-medium text-rose-300 transition-all"
          >
            Clear Buffer
          </button>
        </div>
      </header>

      {/* Metrics Bar */}
      <div className="grid grid-cols-2 sm:grid-cols-6 border-b border-slate-800 bg-slate-950/40 px-6 py-3 gap-4 text-xs font-mono">
        <div>
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Traced Requests</span>
          <span className="text-sm font-semibold text-slate-200">{metrics?.total_requests ?? 0}</span>
        </div>
        <div>
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Median (p50)</span>
          <span className="text-sm font-semibold text-indigo-400">{metrics?.p50_latency_ms ?? 0} ms</span>
        </div>
        <div>
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Tail (p95)</span>
          <span className="text-sm font-semibold text-amber-400">{metrics?.p95_latency_ms ?? 0} ms</span>
        </div>
        <div>
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Avg TTFT</span>
          <span className="text-sm font-semibold text-cyan-400">{metrics?.avg_ttft_ms ?? 0} ms</span>
        </div>
        <div>
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Token Velocity</span>
          <span className="text-sm font-semibold text-emerald-400">{metrics?.avg_tokens_per_second ?? 0} tok/s</span>
        </div>
        <div>
          <span className="text-[10px] text-slate-400 uppercase tracking-wider block">Error Rate</span>
          <span className="text-sm font-semibold text-rose-400">
            {((metrics?.error_rate ?? 0) * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Main Panel */}
      {activeTab === 'waterfall' ? (
        <div className="flex-1 flex overflow-hidden">
          {/* Left Column: Trace List */}
          <div className="w-80 border-r border-slate-800 flex flex-col bg-slate-950/50">
            <div className="p-3 border-b border-slate-850 flex items-center justify-between text-xs text-slate-400 font-medium">
              <span>Recent Traces ({traces.length})</span>
            </div>
            <div className="flex-1 overflow-y-auto divide-y divide-slate-850">
              {traces.length === 0 ? (
                <div className="p-6 text-center text-xs text-slate-500">
                  No traces recorded yet. Send a chat request or tool execution to generate traces.
                </div>
              ) : (
                traces.map((t) => (
                  <button
                    key={t.trace_id}
                    onClick={() => setSelectedTraceId(t.trace_id)}
                    className={`w-full text-left p-3 transition-all flex flex-col gap-1.5 ${
                      selectedTraceId === t.trace_id
                        ? 'bg-indigo-600/15 border-l-2 border-indigo-500'
                        : 'hover:bg-slate-900/60'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-200 truncate max-w-[170px]">
                        {t.root_name}
                      </span>
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-medium ${
                          t.status === 'ERROR'
                            ? 'bg-rose-500/20 text-rose-400'
                            : 'bg-emerald-500/20 text-emerald-400'
                        }`}
                      >
                        {t.total_duration_ms.toFixed(1)} ms
                      </span>
                    </div>
                    <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono">
                      <span>{t.span_count} spans</span>
                      <span>{t.trace_id.slice(0, 8)}...</span>
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Right Column: Waterfall Gantt Chart & Span Inspector */}
          <div className="flex-1 flex flex-col overflow-hidden bg-slate-950">
            {detailLoading ? (
              <div className="flex-1 flex items-center justify-center text-xs text-slate-400">
                Loading trace timeline...
              </div>
            ) : !activeTrace ? (
              <div className="flex-1 flex items-center justify-center text-xs text-slate-500">
                Select a trace from the left panel to inspect its execution waterfall.
              </div>
            ) : (
              <div className="flex-1 flex flex-col overflow-hidden">
                {/* Waterfall Header */}
                <div className="p-4 border-b border-slate-800 bg-slate-900/40 flex items-center justify-between">
                  <div>
                    <h2 className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                      <GitFork className="w-4 h-4 text-indigo-400" />
                      Trace: {activeTrace.root_name}
                    </h2>
                    <span className="text-[11px] font-mono text-slate-400">
                      Trace ID: {activeTrace.trace_id} ({activeTrace.total_duration_ms.toFixed(2)} ms total)
                    </span>
                  </div>
                  <div className="text-xs font-mono text-slate-400">
                    {activeTrace.spans.length} spans captured
                  </div>
                </div>

                {/* Timeline Gantt Chart */}
                <div className="flex-1 overflow-y-auto p-4 space-y-2">
                  <div className="text-[10px] uppercase font-mono text-slate-500 flex justify-between border-b border-slate-850 pb-1">
                    <span>Span Name</span>
                    <span>Timeline (0 ms &rarr; {activeTrace.total_duration_ms.toFixed(1)} ms)</span>
                  </div>

                  {activeTrace.spans.map((s) => {
                    const totalDur = Math.max(1.0, activeTrace.total_duration_ms);
                    const leftPct = Math.min(100, Math.max(0, (s.offset_ms / totalDur) * 100));
                    const widthPct = Math.min(100 - leftPct, Math.max(2, (s.duration_ms / totalDur) * 100));

                    const isChild = s.parent_span_id !== null;
                    const isSelected = selectedSpan?.span_id === s.span_id;

                    return (
                      <div
                        key={s.span_id}
                        onClick={() => setSelectedSpan(s)}
                        className={`flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-all ${
                          isSelected
                            ? 'bg-indigo-950/40 border border-indigo-500/40'
                            : 'hover:bg-slate-900/60 border border-transparent'
                        }`}
                      >
                        {/* Span Name & Indentation */}
                        <div className="w-72 flex items-center gap-2 truncate">
                          {isChild && <span className="text-slate-600 pl-3">&lfloor;</span>}
                          <span className="text-xs font-medium text-slate-200 truncate">{s.name}</span>
                          <span className="text-[10px] font-mono text-slate-500 ml-auto">
                            {s.duration_ms.toFixed(1)}ms
                          </span>
                        </div>

                        {/* Gantt Bar Track */}
                        <div className="flex-1 h-5 bg-slate-900/80 rounded relative overflow-hidden">
                          <div
                            style={{
                              left: `${leftPct}%`,
                              width: `${widthPct}%`,
                            }}
                            className={`absolute top-1 bottom-1 rounded border shadow-sm ${getSpanColor(
                              s.name,
                              s.status
                            )}`}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Selected Span Detail Drawer */}
                {selectedSpan && (
                  <div className="h-56 border-t border-slate-800 bg-slate-900/70 p-4 overflow-y-auto space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-semibold text-slate-100">{selectedSpan.name}</span>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                          {selectedSpan.duration_ms.toFixed(2)} ms
                        </span>
                        <span
                          className={`text-[10px] font-mono px-2 py-0.5 rounded ${
                            selectedSpan.status === 'ERROR'
                              ? 'bg-rose-500/20 text-rose-400'
                              : 'bg-emerald-500/20 text-emerald-400'
                          }`}
                        >
                          {selectedSpan.status}
                        </span>
                      </div>
                      <span className="text-[10px] font-mono text-slate-500">
                        Span ID: {selectedSpan.span_id}
                      </span>
                    </div>

                    {/* OpenTelemetry Attributes */}
                    <div className="space-y-1">
                      <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
                        OpenTelemetry Attributes:
                      </span>
                      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                        {Object.entries(selectedSpan.attributes).map(([k, v]) => (
                          <div key={k} className="p-2 bg-slate-950 rounded border border-slate-850 flex justify-between">
                            <span className="text-indigo-400">{k}:</span>
                            <span className="text-slate-300 truncate max-w-[200px]">{String(v)}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      ) : (
        /* TAB 2: EDUCATIONAL TRACING GUIDE */
        <div className="flex-1 overflow-y-auto p-6 max-w-4xl mx-auto space-y-6 text-xs text-slate-300 leading-relaxed">
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 space-y-4">
            <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-400" />
              Why Distributed Tracing is Critical for LLM Systems
            </h2>
            <p>
              In traditional web applications, requests are usually simple CRUD database queries taking 5-20ms. In modern LLM systems, a single user prompt invokes a multi-hop pipeline that can span 500ms to 5,000ms:
            </p>
            <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono text-[11px] text-indigo-300">
              HTTP Ingress &rarr; Prompt Guard &rarr; Dynamic Router &rarr; Vector RAG Embeddings &rarr; Hybrid BM25 &rarr; LLM Prefill &rarr; Cached Autoregressive Decode (N tokens) &rarr; Secret Redactor
            </div>
            <p>
              When a user reports &quot;the assistant is slow&quot;, standard server logs cannot tell you whether the delay was caused by slow vector embeddings, excessive document chunk re-ranking, token decode throttling, or network streaming latency.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-4">
              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 space-y-2">
                <h3 className="font-semibold text-indigo-400">1. W3C TraceContext Standard</h3>
                <p className="text-slate-400">
                  Headers follow the formal specification <code>00-{'{trace_id}'}-{'{span_id}'}-01</code>, allowing traces to propagate across microservices, frontends, and external LLM APIs seamlessly.
                </p>
              </div>
              <div className="p-4 bg-slate-950 rounded-lg border border-slate-800 space-y-2">
                <h3 className="font-semibold text-emerald-400">2. Latency Waterfall Gantt Charts</h3>
                <p className="text-slate-400">
                  Visual timeline bars immediately spotlight critical-path bottlenecks, revealing exactly which internal span consumed the majority of execution time.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
