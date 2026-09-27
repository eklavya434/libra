'use client';

import React, { useState, useEffect } from 'react';
import { Sparkles, MessageSquare, Terminal, Layers, ArrowRight, X } from 'lucide-react';

interface TourStep {
  title: string;
  description: string;
  targetHint: string;
  icon: React.ElementType;
}

const TOUR_STEPS: TourStep[] = [
  {
    title: 'Start Here: Interactive Assistant',
    description:
      'Chat directly with real LLMs. Conversations and multi-turn states are isolated per session and persisted durably in PostgreSQL.',
    targetHint: 'Core Chat Workspace',
    icon: MessageSquare,
  },
  {
    title: 'The Educational Lab Underneath',
    description:
      'Compare models side-by-side in the Arena, query RAG vector knowledge bases, or test code in the stateful Notebook environment.',
    targetHint: 'Core Lab Modules',
    icon: Terminal,
  },
  {
    title: 'Frontier Research Modules',
    description:
      'Explore 14 advanced research labs: MCTS reasoning, speculative Medusa drafting, sparse MoE, and alignment engines.',
    targetHint: 'Advanced Labs Toggle',
    icon: Layers,
  },
];

export default function OnboardingTour() {
  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    if (typeof window === 'undefined') return;

    // Check if user already dismissed or finished tour
    const alreadyOnboarded = localStorage.getItem('libra_onboarded');
    if (alreadyOnboarded) return;

    // Check visit count: auto-dismiss after 3 visits
    const visitCountStr = localStorage.getItem('libra_visit_count');
    const visitCount = visitCountStr ? parseInt(visitCountStr, 10) : 0;
    const newCount = visitCount + 1;
    localStorage.setItem('libra_visit_count', newCount.toString());

    if (newCount > 3) {
      localStorage.setItem('libra_onboarded', 'true');
      return;
    }

    // Small delay so layout is stable before popping in
    const timer = setTimeout(() => {
      setIsOpen(true);
    }, 800);

    return () => clearTimeout(timer);
  }, []);

  const handleDismiss = () => {
    setIsOpen(false);
    if (typeof window !== 'undefined') {
      localStorage.setItem('libra_onboarded', 'true');
    }
  };

  const handleNext = () => {
    if (currentStep < TOUR_STEPS.length - 1) {
      setCurrentStep((prev) => prev + 1);
    } else {
      handleDismiss();
    }
  };

  if (!isOpen) return null;

  const step = TOUR_STEPS[currentStep];
  const StepIcon = step.icon;

  return (
    <div className="fixed bottom-6 right-6 z-50 max-w-sm rounded-2xl border border-indigo-500/30 bg-slate-950/95 p-5 shadow-2xl shadow-indigo-950/50 backdrop-blur-md animate-in fade-in slide-in-from-bottom-5 duration-300">
      {/* Header */}
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            <StepIcon className="h-4 w-4" />
          </div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-indigo-400">
            Quick Tour ({currentStep + 1} of {TOUR_STEPS.length})
          </span>
        </div>
        <button
          onClick={handleDismiss}
          className="rounded-lg p-1 text-slate-400 hover:bg-white/10 hover:text-white transition-colors"
          title="Dismiss tour"
        >
          <X className="h-4 w-4" />
        </button>
      </div>

      {/* Content */}
      <div className="mt-3">
        <h4 className="text-sm font-semibold text-slate-100">{step.title}</h4>
        <p className="mt-1.5 text-xs leading-relaxed text-slate-300">{step.description}</p>
      </div>

      {/* Footer controls */}
      <div className="mt-4 flex items-center justify-between border-t border-white/[0.07] pt-3">
        <span className="text-[10px] text-slate-500 font-mono">{step.targetHint}</span>
        <div className="flex items-center gap-2">
          <button
            onClick={handleDismiss}
            className="text-xs text-slate-400 hover:text-white px-2 py-1 transition-colors"
          >
            Skip
          </button>
          <button
            onClick={handleNext}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-500 px-3 py-1.5 text-xs font-medium text-white hover:bg-indigo-400 transition-colors shadow-sm"
          >
            <span>{currentStep === TOUR_STEPS.length - 1 ? 'Got it' : 'Next'}</span>
            <ArrowRight className="h-3 w-3" />
          </button>
        </div>
      </div>
    </div>
  );
}
