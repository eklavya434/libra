'use client';

import React, { useEffect, useState, useCallback, useRef } from 'react';
import Link from 'next/link';
import {
  MessageSquare,
  FlaskConical,
  Gauge,
  Search,
  NotebookPen,
  ShieldCheck,
  ArrowRight,
  Activity,
  RefreshCw,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react';
import { apiPath } from '@/lib/api';
import { LANDING_FEATURES_STATUS, getMaturityStyle } from '@/lib/module-status';

type BackendStatus = 'idle' | 'connecting' | 'slow' | 'connected' | 'error';

interface HealthState {
  status: BackendStatus;
  checks: { name: string; status: string }[];
}

const features = [
  {
    icon: MessageSquare,
    title: 'Chat Assistant',
    description:
      'A ChatGPT-like assistant that answers through configured real model providers — or runs entirely on-device in the educational lab.',
  },
  {
    icon: FlaskConical,
    title: 'Transformer Lab',
    description:
      'Learn attention, tokens, training loops and checkpoints by building a tiny transformer from scratch inside the notebook.',
  },
  {
    icon: Gauge,
    title: 'Arena & Evaluation',
    description:
      'Head-to-head model comparisons, Elo leaderboards, perplexity and latency metrics for every provider in the registry.',
  },
  {
    icon: Search,
    title: 'RAG Knowledge Base',
    description:
      'Upload documents, chunk them, and retrieve answers with hybrid dense + BM25 retrieval and reranking.',
  },
  {
    icon: NotebookPen,
    title: 'Stateful Data Notebook',
    description:
      'In-browser analytics kernels with charts, tables and persistent state for exploring data without a cloud GPU.',
  },
  {
    icon: ShieldCheck,
    title: 'Security & Honesty First',
    description:
      'Guest-first design, rate limiting, secret scanning, no fabricated metrics — and every mock explicitly labelled.',
  },
];

export default function LandingPage() {
  const [health, setHealth] = useState<HealthState>({ status: 'connecting', checks: [] });
  const slowTimerRef = useRef<NodeJS.Timeout | null>(null);

  const checkHealth = useCallback(async () => {
    setHealth((prev) => ({ ...prev, status: 'connecting' }));

    // Trigger 'slow' state if connecting takes more than 5 seconds (Render cold-start)
    if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
    slowTimerRef.current = setTimeout(() => {
      setHealth((prev) => (prev.status === 'connecting' ? { ...prev, status: 'slow' } : prev));
    }, 5000);

    try {
      // Short fast probe first (2.5s) to check if already warm
      let res = await fetch(apiPath('/healthz'), {
        headers: { Accept: 'application/json' },
        signal: AbortSignal.timeout(2500),
      }).catch(() => null);

      // If fast probe didn't resolve, perform deeper health check with 40s timeout for cold start
      if (!res || !res.ok) {
        res = await fetch(apiPath('/api/v1/health'), {
          headers: { Accept: 'application/json' },
          signal: AbortSignal.timeout(40000),
        }).catch(() => null);
      }

      if (slowTimerRef.current) clearTimeout(slowTimerRef.current);

      if (res && res.ok) {
        const body: { checks?: { name: string; status: string }[] } = await res.json().catch(() => ({}));
        setHealth({ status: 'connected', checks: body.checks ?? [] });
      } else {
        setHealth({ status: 'error', checks: [] });
      }
    } catch {
      if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
      setHealth({ status: 'error', checks: [] });
    }
  }, []);

  useEffect(() => {
    checkHealth();
    return () => {
      if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
    };
  }, [checkHealth]);

  return (
    <div className="min-h-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* Hero Header */}
      <header className="libra-fade-in relative z-10 mx-auto flex max-w-7xl items-center justify-between px-6 py-6 lg:px-8">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-indigo-400/20 bg-gradient-to-br from-indigo-400 to-violet-600 font-bold text-white shadow-lg shadow-indigo-500/20">
            L
          </div>
          <span className="text-lg font-semibold tracking-tight">Libra</span>
        </div>
        <Link
          href="/chat"
          className="libra-interactive rounded-xl border border-white/10 bg-white/[0.06] px-4 py-2 text-sm font-medium text-slate-100 shadow-lg shadow-black/10 hover:border-indigo-400/30 hover:bg-indigo-500/15"
        >
          Open the Assistant
        </Link>
      </header>

      <main className="relative mx-auto max-w-7xl px-6 lg:px-8">
        <section className="relative flex flex-col items-center py-20 text-center lg:py-28">
          {/* Backend Status State Machine */}
          <div className="libra-fade-up mb-6">
            {health.status === 'connecting' && (
              <div className="inline-flex items-center gap-2 rounded-full border border-indigo-500/30 bg-indigo-500/10 px-4 py-1.5 text-xs text-indigo-300 shadow-xl shadow-black/10">
                <RefreshCw className="h-3.5 w-3.5 animate-spin text-indigo-400" />
                <span>Checking backend connection…</span>
              </div>
            )}

            {health.status === 'slow' && (
              <div className="inline-flex items-center gap-2.5 rounded-full border border-amber-500/30 bg-amber-500/10 px-4 py-1.5 text-xs text-amber-300 shadow-xl shadow-black/10">
                <RefreshCw className="h-3.5 w-3.5 animate-spin text-amber-400" />
                <span>Waking up the backend — this can take up to 30s on the free tier. Hang tight.</span>
              </div>
            )}

            {health.status === 'connected' && (
              <div className="inline-flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-1.5 text-xs text-emerald-300 shadow-xl shadow-black/10">
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400" />
                <span>Backend online</span>
              </div>
            )}

            {health.status === 'error' && (
              <div className="inline-flex flex-wrap items-center justify-center gap-2 rounded-xl border border-amber-500/40 bg-amber-950/30 px-4 py-2 text-xs text-amber-300 shadow-xl shadow-black/10">
                <AlertCircle className="h-4 w-4 text-amber-400 shrink-0" />
                <span>The backend may be asleep or redeploying — click retry:</span>
                <button
                  onClick={checkHealth}
                  className="ml-1 inline-flex items-center gap-1 rounded-lg bg-amber-500/20 px-2.5 py-1 font-semibold text-amber-200 hover:bg-amber-500/30 hover:text-white transition-colors"
                >
                  <RefreshCw className="h-3 w-3" />
                  Retry
                </button>
              </div>
            )}
          </div>

          <h1 className="libra-fade-up libra-stagger-1 max-w-4xl bg-gradient-to-b from-white via-white to-slate-400 bg-clip-text text-4xl font-semibold leading-[1.04] tracking-[-0.035em] text-transparent sm:text-6xl lg:text-7xl">
            An AI assistant and LLM laboratory, built from first principles.
          </h1>
          <p className="libra-fade-up libra-stagger-2 mt-7 max-w-2xl text-base leading-7 text-slate-400 sm:text-lg">
            Libra is a ChatGPT-like assistant with an educational transformer lab underneath —
            every component of the stack, from tokenization to attention to training, is
            explainable and inspectable.
          </p>

          {/* Primary vs Secondary CTA Hierarchy */}
          <div className="libra-fade-up libra-stagger-3 mt-10 flex flex-wrap items-center justify-center gap-4">
            <Link
              href="/chat"
              className="libra-interactive inline-flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-indigo-500 via-indigo-600 to-violet-600 px-7 py-3.5 text-base font-semibold text-white shadow-xl shadow-indigo-500/25 hover:from-indigo-400 hover:to-violet-500 hover:shadow-indigo-500/40 transition-all group"
            >
              Start chatting — no sign-up needed
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <a
              href="https://github.com/eklavya434/libra"
              target="_blank"
              rel="noopener noreferrer"
              className="libra-interactive inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.02] px-5 py-3.5 text-sm font-medium text-slate-300 hover:border-white/20 hover:bg-white/[0.06] hover:text-white transition-all"
            >
              View the source
            </a>
          </div>
        </section>

        {/* Feature Grid & Maturity Framing */}
        <section className="pb-24">
          <div className="mb-8 flex flex-col items-center text-center">
            <p className="text-xs uppercase tracking-wider text-indigo-400 font-semibold">Capabilities & Status</p>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-slate-400">
              Libra ships in phases — every card below is marked with what&apos;s live today versus what&apos;s actively being built. Nothing here is simulated without being labelled as such.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {features.map((feature) => {
              const maturity = LANDING_FEATURES_STATUS[feature.title] || 'stable';
              const style = getMaturityStyle(maturity);
              return (
                <div
                  key={feature.title}
                  className="libra-fade-up libra-hover-lift libra-surface group relative flex flex-col justify-between rounded-2xl p-6"
                >
                  <div>
                    <div className="mb-5 flex items-center justify-between">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-indigo-400/15 bg-indigo-500/10 text-indigo-300 transition-transform duration-300 group-hover:scale-105">
                        <feature.icon className="h-5 w-5" />
                      </div>
                      <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-medium border ${style.bgColor} ${style.textColor} ${style.borderColor}`}>
                        <span className={`h-1.5 w-1.5 rounded-full ${style.dotColor}`} />
                        {style.badgeText}
                      </span>
                    </div>
                    <h3 className="text-lg font-semibold">{feature.title}</h3>
                    <p className="mt-2 text-sm leading-relaxed text-slate-400">{feature.description}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Transparency status */}
        <section className="libra-scale-in libra-surface mb-24 rounded-2xl p-8 sm:p-10">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-indigo-400" />
            <h2 className="text-xl font-semibold">Operational transparency</h2>
          </div>
          <p className="mt-2 max-w-2xl text-sm text-slate-400">
            Libra runs on a strict zero-cost philosophy. Live inference uses real configured
            providers; anything simulated is clearly labelled. Status is always reported honestly —
            never fabricated.
          </p>
          {health.checks.length > 0 && (
            <ul className="mt-4 max-w-2xl space-y-1">
              {health.checks.map((check) => (
                <li key={check.name} className="flex items-center gap-2 text-sm text-slate-300">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      check.status === 'ok' ? 'bg-emerald-400' : 'bg-amber-400'
                    }`}
                  />
                  {check.name}
                  {check.status === 'ok' ? ' ✓' : ` (${check.status})`}
                </li>
              ))}
            </ul>
          )}
        </section>
      </main>

      <footer className="border-t border-white/[0.07] py-8">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 text-sm text-slate-500">
          <span>Libra — educational LLM laboratory &amp; assistant.</span>
          <span>Built from first principles, on a CPU budget.</span>
        </div>
      </footer>
    </div>
  );
}