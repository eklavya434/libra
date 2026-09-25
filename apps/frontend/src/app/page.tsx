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
    <div className="min-h-screen bg-slate-950 text-slate-100">
      {/* Hero */}
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-6">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-500 font-bold text-white">
            L
          </div>
          <span className="text-lg font-semibold tracking-tight">Libra</span>
        </div>
        <Link
          href="/chat"
          className="rounded-lg bg-indigo-500 px-4 py-2 text-sm font-medium text-white transition hover:bg-indigo-400"
        >
          Open the Assistant
        </Link>
      </header>

      <main className="mx-auto max-w-6xl px-6">
        <section className="flex flex-col items-center py-20 text-center">
          <p className="mb-4 inline-flex items-center gap-1.5 rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-400">
            <Activity className="h-3.5 w-3.5" />
            {health.online === null && <span>Checking backend…</span>}
            {health.online === true && <span>Backend online</span>}
            {health.online === false && <span className="text-amber-400">Backend unreachable</span>}
          </p>
          <h1 className="max-w-3xl text-4xl font-bold leading-tight tracking-tight sm:text-6xl">
            An AI assistant and LLM laboratory, built from first principles.
          </h1>
          <p className="mt-6 max-w-2xl text-lg text-slate-400">
            Libra is a ChatGPT-like assistant with an educational transformer lab underneath —
            every component of the stack, from tokenization to attention to training, is
            explainable and inspectable.
          </p>
          <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
            <Link
              href="/chat"
              className="inline-flex items-center gap-2 rounded-lg bg-indigo-500 px-6 py-3 text-sm font-semibold text-white transition hover:bg-indigo-400"
            >
              Start chatting — no sign-up needed
              <ArrowRight className="h-4 w-4" />
            </Link>
            <a
              href="https://github.com/eklavya434/libra"
              target="_blank"
              rel="noopener noreferrer"
              className="rounded-lg border border-slate-700 px-6 py-3 text-sm font-medium text-slate-300 transition hover:border-slate-500"
            >
              View the source
            </a>
          </div>
        </section>

        {/* Feature grid */}
        <section className="grid grid-cols-1 gap-6 pb-20 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="rounded-2xl border border-slate-800 bg-slate-900/50 p-6 transition hover:border-slate-600"
            >
              <feature.icon className="h-6 w-6 text-indigo-400" />
              <h3 className="mt-4 text-lg font-semibold">{feature.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-slate-400">{feature.description}</p>
            </div>
          ))}
        </section>

        {/* Transparency status */}
        <section className="mb-20 rounded-2xl border border-slate-800 bg-slate-900/50 p-8">
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

      <footer className="border-t border-slate-800 py-8">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 text-sm text-slate-500">
          <span>Libra — educational LLM laboratory &amp; assistant.</span>
          <span>Built from first principles, on a CPU budget.</span>
        </div>
      </footer>
    </div>
  );
}