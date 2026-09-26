'use client';

import React, { useEffect, useState } from 'react';
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
} from 'lucide-react';
import { apiPath } from '@/lib/api';

interface HealthState {
  online: boolean | null;
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
  const [health, setHealth] = useState<HealthState>({ online: null, checks: [] });

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(apiPath('/api/v1/health'), {
          headers: { Accept: 'application/json' },
          signal: AbortSignal.timeout(8000),
        });
        if (!res.ok) throw new Error(`health ${res.status}`);
        const body: { checks?: { name: string; status: string }[] } = await res.json();
        if (!cancelled) {
          setHealth({ online: true, checks: body.checks ?? [] });
        }
      } catch {
        if (!cancelled) {
          setHealth({ online: false, checks: [] });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="min-h-screen overflow-hidden bg-slate-950 text-slate-100">
      {/* Hero */}
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
        <section className="relative flex flex-col items-center py-24 text-center lg:py-32">
          <p className="libra-fade-up inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.035] px-3.5 py-1.5 text-xs text-slate-400 shadow-xl shadow-black/10">
            <Activity className="h-3.5 w-3.5" />
            {health.online === null && <span>Checking backend…</span>}
            {health.online === true && <span>Backend online</span>}
            {health.online === false && <span className="text-amber-400">Backend unreachable</span>}
          </p>
          <h1 className="libra-fade-up libra-stagger-1 max-w-4xl bg-gradient-to-b from-white via-white to-slate-400 bg-clip-text text-4xl font-semibold leading-[1.04] tracking-[-0.035em] text-transparent sm:text-6xl lg:text-7xl">
            An AI assistant and LLM laboratory, built from first principles.
          </h1>
          <p className="libra-fade-up libra-stagger-2 mt-7 max-w-2xl text-base leading-7 text-slate-400 sm:text-lg">
            Libra is a ChatGPT-like assistant with an educational transformer lab underneath —
            every component of the stack, from tokenization to attention to training, is
            explainable and inspectable.
          </p>
          <div className="libra-fade-up libra-stagger-3 mt-10 flex flex-wrap items-center justify-center gap-3">
            <Link
              href="/chat"
              className="libra-interactive inline-flex items-center gap-2 rounded-xl bg-indigo-500 px-6 py-3 text-sm font-semibold text-white shadow-xl shadow-indigo-950/40 hover:bg-indigo-400"
            >
              Start chatting — no sign-up needed
              <ArrowRight className="h-4 w-4" />
            </Link>
            <a
              href="https://github.com/eklavya434/libra"
              target="_blank"
              rel="noopener noreferrer"
              className="libra-interactive rounded-xl border border-white/10 bg-white/[0.025] px-6 py-3 text-sm font-medium text-slate-300 hover:border-white/20 hover:bg-white/[0.05]"
            >
              View the source
            </a>
          </div>
        </section>

        {/* Feature grid */}
        <section className="grid grid-cols-1 gap-4 pb-24 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="libra-fade-up libra-hover-lift libra-surface group rounded-2xl p-6"
            >
              <div className="mb-5 flex h-10 w-10 items-center justify-center rounded-xl border border-indigo-400/15 bg-indigo-500/10 text-indigo-300 transition-transform duration-300 group-hover:scale-105">
              <feature.icon className="h-5 w-5" />
            </div>
              <h3 className="mt-4 text-lg font-semibold">{feature.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{feature.description}</p>
            </div>
          ))}
        </section>

        {/* Transparency status */}
        <section className="libra-scale-in libra-surface mb-24 rounded-2xl p-8 sm:p-10">
          <h2 className="text-xl font-semibold">Operational transparency</h2>
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