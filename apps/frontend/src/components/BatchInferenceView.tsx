'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Layers,
  Play,
  RotateCcw,
  CheckCircle2,
  XCircle,
  Clock,
  Zap,
  TrendingUp,
  BarChart3,
  ListOrdered,
  BookOpen,
} from 'lucide-react';

interface BatchJobItem {
  index: number;
  prompt: string;
  completion: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  status: string;
}

interface BatchJobTelemetry {
  total_prompts: number;
  completed_prompts: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  wall_clock_time_sec: number;
  throughput_tokens_per_sec: number;
  padding_waste_pct: number;
  speedup_factor: number;
  num_buckets: number;
}

interface BatchJobSummary {
  job_id: string;
  name: string;
  model_name: string;
  num_prompts: number;
  max_new_tokens: number;
  temperature: number;
  binning_strategy: string;
  priority: string;
  status: string;
  created_at: number;
  started_at: number | null;
  finished_at: number | null;
  progress: number;
  current_bucket: number;
  total_buckets: number;
  results: BatchJobItem[];
  telemetry: BatchJobTelemetry;
  error_message: string | null;
}

const PRESET_WORKLOADS = [
  {
    name: 'Sentiment Analysis Batch (12 Prompts)',
    description: 'Variable length customer feedback texts classified in high-throughput batches.',
    prompts: [
      'Amazing product, arrived on time and exceeded expectations!',
      'Terrible service. The package was damaged and customer support was completely unhelpful.',
      'Neutral experience.',
      'I was hesitant at first because of the price, but after using it for two weeks in my daily workflow, it is well worth every single penny.',
      'Super fast shipping!',
      'Defective on arrival, requesting an immediate refund.',
      'Decent quality for the price point.',
      'The battery life is slightly shorter than advertised, but the screen and build quality are phenomenal for this price tier.',
      'Five stars!',
      'Not worth it.',
      'Exceptional customer support team resolved my query in under ten minutes.',
      'Would buy again.',
    ],
  },
  {
    name: 'Code Docstring Generation (8 Functions)',
    description: 'Disparate Python code snippets needing docstrings, showing extreme length variance.',
    prompts: [
      'def add(a, b): return a + b',
      'def calculate_shannon_entropy(text: str) -> float:\n    counts = Counter(text)\n    n = len(text)\n    return -sum((c / n) * math.log2(c / n) for c in counts.values())',
      'def identity(x): return x',
      'class RingBuffer:\n    def __init__(self, cap=100):\n        self.buf = [None] * cap\n        self.head = 0\n    def append(self, item):\n        self.buf[self.head] = item\n        self.head = (self.head + 1) % len(self.buf)',
      'def is_even(n): return n % 2 == 0',
      'def clip_grad_norm(params, max_norm):\n    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in params if p.grad is not None]), 2)\n    clip_coef = max_norm / (total_norm + 1e-6)\n    if clip_coef < 1:\n        for p in params: p.grad.detach().mul_(clip_coef)',
      'def ping(): return "pong"',
      'def parse_w3c_traceparent(header: str) -> dict:\n    parts = header.strip().split("-")\n    return {"trace_id": parts[1], "span_id": parts[2]}',
    ],
  },
];

export default function BatchInferenceView() {
  const [activeTab, setActiveTab] = useState<'submitter' | 'visualizer' | 'edu'>('submitter');
  const [jobs, setJobs] = useState<BatchJobSummary[]>([]);
  const [selectedJob, setSelectedJob] = useState<BatchJobSummary | null>(null);

  // Form state
  const [jobName, setJobName] = useState(PRESET_WORKLOADS[0].name);
  const [promptText, setPromptText] = useState(PRESET_WORKLOADS[0].prompts.join('\n'));
  const [binningStrategy, setBinningStrategy] = useState('dynamic_length');
  const [maxTokens, setMaxTokens] = useState(16);
  const [priority, setPriority] = useState('NORMAL');
  const [submitting, setSubmitting] = useState(false);

  // Analysis state
  const [analysisResult, setAnalysisResult] = useState<any>(null);

  const fetchJobs = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/batch/jobs');
      if (res.ok) {
        const data = await res.json();
        setJobs(data.jobs || []);
        if (data.jobs && data.jobs.length > 0 && !selectedJob) {
          setSelectedJob(data.jobs[0]);
        }
      }
    } catch (err) {
      console.error('Failed to fetch batch jobs:', err);
    }
  }, [selectedJob]);

  useEffect(() => {
    fetchJobs();
    const interval = setInterval(fetchJobs, 3000);
    return () => clearInterval(interval);
  }, [fetchJobs]);

  const runAnalysis = useCallback(async () => {
    try {
      const prompts = promptText.split('\n').map((s) => s.trim()).filter(Boolean);
      if (prompts.length === 0) return;
      const res = await fetch('/api/v1/batch/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompts,
          max_batch_size: 4,
          max_tokens_per_bucket: 256,
        }),
      });
      if (res.ok) {
        const data = await res.json();
        setAnalysisResult(data);
      }
    } catch (err) {
      console.error('Failed to run batch analysis:', err);
    }
  }, [promptText]);

  useEffect(() => {
    runAnalysis();
  }, [runAnalysis]);

  const handleSubmitJob = async () => {
    const prompts = promptText.split('\n').map((s) => s.trim()).filter(Boolean);
    if (prompts.length === 0) return;

    try {
      setSubmitting(true);
      const res = await fetch('/api/v1/batch/jobs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: jobName,
          prompts,
          max_new_tokens: maxTokens,
          binning_strategy: binningStrategy,
          priority,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setSelectedJob(data.job);
        fetchJobs();
      }
    } catch (err) {
      console.error('Failed to submit batch job:', err);
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancelJob = async (jobId: string) => {
    try {
      await fetch(`/api/v1/batch/jobs/${jobId}`, { method: 'DELETE' });
      fetchJobs();
    } catch (err) {
      console.error('Failed to cancel job:', err);
    }
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-100 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-slate-800/80 bg-slate-900/40 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-amber-500 to-orange-500 flex items-center justify-center font-bold text-white shadow-lg shadow-amber-500/20">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-bold text-lg text-slate-100">High-Throughput Batch Inference & Workers</h2>
              <span className="text-[11px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/30 font-mono">
                Phase 39
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Dynamic sequence length binning, decoder left-padding, and priority worker queues.
            </p>
          </div>
        </div>

        {/* Tab switcher */}
        <div className="flex items-center bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
          <button
            onClick={() => setActiveTab('submitter')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
              activeTab === 'submitter'
                ? 'bg-amber-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Play className="w-3.5 h-3.5" />
            <span>Job Runner & Queue</span>
          </button>
          <button
            onClick={() => setActiveTab('visualizer')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
              activeTab === 'visualizer'
                ? 'bg-amber-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            <span>Sequence Binning Visualizer</span>
          </button>
          <button
            onClick={() => setActiveTab('edu')}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-medium transition-all ${
              activeTab === 'edu'
                ? 'bg-amber-600 text-white shadow-sm'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            <span>Architectural Guide</span>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
        {/* Metric Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 flex flex-col">
            <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5">
              <ListOrdered className="w-3.5 h-3.5 text-amber-400" />
              Total Batch Jobs
            </span>
            <span className="text-2xl font-bold text-slate-100 mt-2 font-mono">{jobs.length}</span>
            <span className="text-[11px] text-slate-400 mt-1">Processed in active session</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 flex flex-col">
            <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-blue-400" />
              Worker Concurrency
            </span>
            <span className="text-2xl font-bold text-blue-400 mt-2 font-mono">1 Worker (CPU)</span>
            <span className="text-[11px] text-slate-400 mt-1">Zero-cost CPU thread pool</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 flex flex-col">
            <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5">
              <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
              Padding Waste Reduction
            </span>
            <span className="text-2xl font-bold text-emerald-400 mt-2 font-mono">
              {analysisResult ? `${analysisResult.efficiency_gain_pct}%` : '50-80%'}
            </span>
            <span className="text-[11px] text-slate-400 mt-1">Saved vs naive batching</span>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800/80 flex flex-col">
            <span className="text-xs font-medium text-slate-400 flex items-center gap-1.5">
              <Zap className="w-3.5 h-3.5 text-purple-400" />
              Estimated FLOP Speedup
            </span>
            <span className="text-2xl font-bold text-purple-400 mt-2 font-mono">
              {analysisResult ? `${analysisResult.estimated_speedup}x` : '1.8x'}
            </span>
            <span className="text-[11px] text-slate-400 mt-1">Quadratic attention acceleration</span>
          </div>
        </div>

        {/* TAB 1: SUBMITTER & QUEUE */}
        {activeTab === 'submitter' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left: Submission Form (5 cols) */}
            <div className="lg:col-span-5 space-y-4">
              <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-4">
                <h3 className="font-semibold text-sm text-slate-200 flex items-center gap-2">
                  <Play className="w-4 h-4 text-amber-400" />
                  Submit New Batch Workload
                </h3>

                {/* Presets */}
                <div>
                  <label className="text-xs text-slate-400 block mb-1.5">Workload Preset</label>
                  <div className="grid grid-cols-1 gap-2">
                    {PRESET_WORKLOADS.map((p, idx) => (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => {
                          setJobName(p.name);
                          setPromptText(p.prompts.join('\n'));
                        }}
                        className="text-left p-2.5 rounded-lg border border-slate-800 hover:border-amber-500/50 hover:bg-slate-850 transition-all text-xs"
                      >
                        <div className="font-medium text-slate-200">{p.name}</div>
                        <div className="text-[11px] text-slate-400 truncate mt-0.5">{p.description}</div>
                      </button>
                    ))}
                  </div>
                </div>

                {/* Job Name */}
                <div>
                  <label className="text-xs text-slate-400 block mb-1.5">Job Name</label>
                  <input
                    type="text"
                    value={jobName}
                    onChange={(e) => setJobName(e.target.value)}
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-amber-500/80"
                  />
                </div>

                {/* Prompts Input */}
                <div>
                  <div className="flex justify-between items-center mb-1.5">
                    <label className="text-xs text-slate-400">Prompts (one per line)</label>
                    <span className="text-[10px] text-slate-500 font-mono">
                      {promptText.split('\n').filter((s) => s.trim()).length} prompts
                    </span>
                  </div>
                  <textarea
                    rows={6}
                    value={promptText}
                    onChange={(e) => setPromptText(e.target.value)}
                    placeholder="Enter one prompt per line..."
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg p-3 text-xs font-mono text-slate-200 focus:outline-none focus:border-amber-500/80 resize-none"
                  />
                </div>

                {/* Controls */}
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Binning</label>
                    <select
                      value={binningStrategy}
                      onChange={(e) => setBinningStrategy(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 focus:outline-none focus:border-amber-500"
                    >
                      <option value="dynamic_length">Dynamic Length</option>
                      <option value="fixed_buckets">Fixed Buckets</option>
                      <option value="naive">Naive Batch</option>
                    </select>
                  </div>

                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Max Tokens</label>
                    <input
                      type="number"
                      value={maxTokens}
                      min={8}
                      max={128}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value) || 16)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 focus:outline-none focus:border-amber-500 font-mono"
                    />
                  </div>

                  <div>
                    <label className="text-[11px] text-slate-400 block mb-1">Priority</label>
                    <select
                      value={priority}
                      onChange={(e) => setPriority(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-800 rounded-lg p-2 text-xs text-slate-200 focus:outline-none focus:border-amber-500"
                    >
                      <option value="HIGH">High (1)</option>
                      <option value="NORMAL">Normal (2)</option>
                      <option value="LOW">Low (3)</option>
                    </select>
                  </div>
                </div>

                {/* Submit button */}
                <button
                  onClick={handleSubmitJob}
                  disabled={submitting}
                  className="w-full flex items-center justify-center gap-2 bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white font-medium py-2.5 px-4 rounded-lg text-xs transition-all shadow-md shadow-amber-600/20 disabled:opacity-50"
                >
                  {submitting ? <RotateCcw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  <span>Enqueue Batch Job</span>
                </button>
              </div>
            </div>

            {/* Right: Active Job Detail & Queue List (7 cols) */}
            <div className="lg:col-span-7 space-y-4">
              {/* Active Selected Job Display */}
              {selectedJob ? (
                <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <h4 className="font-semibold text-slate-200">{selectedJob.name}</h4>
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                          {selectedJob.job_id}
                        </span>
                      </div>
                      <span className="text-xs text-slate-400">
                        {selectedJob.num_prompts} prompts • {selectedJob.binning_strategy} • Priority: {selectedJob.priority}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span
                        className={`text-xs px-2.5 py-1 rounded-full font-medium flex items-center gap-1.5 ${
                          selectedJob.status === 'completed'
                            ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                            : selectedJob.status === 'processing'
                            ? 'bg-amber-500/20 text-amber-300 border border-amber-500/30'
                            : selectedJob.status === 'cancelled'
                            ? 'bg-red-500/20 text-red-300 border border-red-500/30'
                            : 'bg-blue-500/20 text-blue-300 border border-blue-500/30'
                        }`}
                      >
                        {selectedJob.status === 'completed' && <CheckCircle2 className="w-3.5 h-3.5" />}
                        {selectedJob.status === 'processing' && <RotateCcw className="w-3.5 h-3.5 animate-spin" />}
                        {selectedJob.status === 'cancelled' && <XCircle className="w-3.5 h-3.5" />}
                        <span className="capitalize">{selectedJob.status}</span>
                      </span>

                      {selectedJob.status === 'processing' && (
                        <button
                          onClick={() => handleCancelJob(selectedJob.job_id)}
                          className="text-xs text-red-400 hover:text-red-300 bg-red-950/30 border border-red-800/40 px-2.5 py-1 rounded-lg"
                        >
                          Cancel
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs text-slate-400 font-mono">
                      <span>Progress: {Math.round(selectedJob.progress * 100)}%</span>
                      {selectedJob.total_buckets > 0 && (
                        <span>
                          Bucket {selectedJob.current_bucket} / {selectedJob.total_buckets}
                        </span>
                      )}
                    </div>
                    <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-amber-500 to-orange-500 transition-all duration-300"
                        style={{ width: `${Math.max(3, selectedJob.progress * 100)}%` }}
                      />
                    </div>
                  </div>

                  {/* Telemetry Strip */}
                  {selectedJob.telemetry && (
                    <div className="grid grid-cols-4 gap-2 pt-2 border-t border-slate-800 text-xs font-mono">
                      <div>
                        <div className="text-slate-500 text-[10px]">THROUGHPUT</div>
                        <div className="text-slate-200 font-bold">
                          {selectedJob.telemetry.throughput_tokens_per_sec.toFixed(1)} tok/s
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-500 text-[10px]">WALL TIME</div>
                        <div className="text-slate-200 font-bold">
                          {selectedJob.telemetry.wall_clock_time_sec.toFixed(2)}s
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-500 text-[10px]">PADDING WASTE</div>
                        <div className="text-amber-400 font-bold">
                          {selectedJob.telemetry.padding_waste_pct.toFixed(1)}%
                        </div>
                      </div>
                      <div>
                        <div className="text-slate-500 text-[10px]">SPEEDUP</div>
                        <div className="text-emerald-400 font-bold">
                          {selectedJob.telemetry.speedup_factor.toFixed(2)}x
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Results Items Table */}
                  <div className="space-y-2 pt-2 border-t border-slate-800">
                    <span className="text-xs font-medium text-slate-300 block">
                      Generated Batch Items ({selectedJob.results.length})
                    </span>
                    <div className="max-h-56 overflow-y-auto space-y-2 pr-1">
                      {selectedJob.results.map((item) => (
                        <div
                          key={item.index}
                          className="p-3 rounded-lg bg-slate-950/80 border border-slate-850 text-xs space-y-1.5"
                        >
                          <div className="flex items-center justify-between text-slate-400 font-mono text-[11px]">
                            <span className="text-amber-400">#{item.index + 1} Prompt</span>
                            <span>{item.latency_ms.toFixed(1)}ms • {item.prompt_tokens}p + {item.completion_tokens}c tok</span>
                          </div>
                          <div className="text-slate-300 italic truncate font-sans">
                            &ldquo;{item.prompt}&rdquo;
                          </div>
                          <div className="text-emerald-300 bg-slate-900/90 p-2 rounded border border-slate-800 font-mono text-[11px]">
                            {item.completion}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-8 rounded-xl bg-slate-900/40 border border-slate-800/60 text-center text-xs text-slate-500">
                  Select or submit a batch job to view real-time execution.
                </div>
              )}

              {/* Historical Queue Table */}
              <div className="p-5 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-3">
                <h4 className="font-semibold text-xs text-slate-300">Recent Job Queue</h4>
                <div className="max-h-48 overflow-y-auto space-y-1.5">
                  {jobs.map((j) => (
                    <div
                      key={j.job_id}
                      onClick={() => setSelectedJob(j)}
                      className={`p-2.5 rounded-lg border cursor-pointer transition-all flex items-center justify-between text-xs ${
                        selectedJob?.job_id === j.job_id
                          ? 'bg-amber-600/10 border-amber-500/40 text-amber-200'
                          : 'bg-slate-950/60 border-slate-850 text-slate-300 hover:bg-slate-900'
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[10px] text-slate-500">{j.job_id}</span>
                        <span className="font-medium truncate max-w-[180px]">{j.name}</span>
                      </div>
                      <div className="flex items-center gap-3 text-[11px] font-mono">
                        <span className="text-slate-400">{j.num_prompts} items</span>
                        <span className="capitalize">{j.status}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: SEQUENCE BINNING VISUALIZER */}
        {activeTab === 'visualizer' && (
          <div className="space-y-6">
            <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-4">
              <div>
                <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-amber-400" />
                  Dynamic Length Binning vs Naive Uniform Batching
                </h3>
                <p className="text-xs text-slate-400 mt-1">
                  In naive batching, every sequence is padded to the longest prompt in the batch (L_max), wasting GPU/CPU FLOPs.
                  Dynamic sequence binning clusters prompts of similar lengths together.
                </p>
              </div>

              {analysisResult && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
                  {/* Naive Batch Visualization */}
                  <div className="p-4 rounded-xl bg-slate-950 border border-red-900/30 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="font-semibold text-xs text-red-400">1. Naive Batching (Unsorted)</span>
                      <span className="text-xs font-mono text-red-400 bg-red-950/40 px-2 py-0.5 rounded border border-red-800/50">
                        {analysisResult.naive.waste_pct}% Padding Waste
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Total Tokens: <span className="font-mono text-slate-200">{analysisResult.naive.total_tokens}</span> • 
                      Padding Waste: <span className="font-mono text-red-300">{analysisResult.naive.padding_tokens} tokens</span>
                    </p>

                    {/* Naive Visual Matrix */}
                    <div className="space-y-1.5 p-3 rounded-lg bg-slate-900/80 border border-slate-800">
                      {[15, 25, 45, 140].map((len, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-[10px] font-mono">
                          <span className="w-12 text-slate-500">#{idx + 1} ({len}t)</span>
                          <div className="flex-1 flex h-4 rounded overflow-hidden bg-slate-800">
                            <div className="bg-blue-600 flex items-center justify-center text-[9px] text-white" style={{ width: `${(len / 140) * 100}%` }}>
                              Real
                            </div>
                            <div className="bg-red-900/60 flex items-center justify-center text-[9px] text-red-300" style={{ width: `${((140 - len) / 140) * 100}%` }}>
                              Waste ({140 - len} pad)
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Dynamic Binning Visualization */}
                  <div className="p-4 rounded-xl bg-slate-950 border border-emerald-900/30 space-y-3">
                    <div className="flex justify-between items-center">
                      <span className="font-semibold text-xs text-emerald-400">2. Dynamic Sequence Binning</span>
                      <span className="text-xs font-mono text-emerald-400 bg-emerald-950/40 px-2 py-0.5 rounded border border-emerald-800/50">
                        {analysisResult.binned.waste_pct}% Padding Waste
                      </span>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Total Tokens: <span className="font-mono text-slate-200">{analysisResult.binned.total_tokens}</span> • 
                      Padding Waste: <span className="font-mono text-emerald-300">{analysisResult.binned.padding_tokens} tokens</span>
                    </p>

                    {/* Binned Visual Matrix */}
                    <div className="space-y-2 p-3 rounded-lg bg-slate-900/80 border border-slate-800">
                      <div className="text-[10px] text-amber-400 font-mono">Bucket #1 (Short Sequences: max 25t)</div>
                      {[15, 25].map((len, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-[10px] font-mono">
                          <span className="w-12 text-slate-500">#{idx + 1} ({len}t)</span>
                          <div className="flex-1 flex h-4 rounded overflow-hidden bg-slate-800">
                            <div className="bg-blue-600 flex items-center justify-center text-[9px] text-white" style={{ width: `${(len / 25) * 100}%` }}>
                              Real
                            </div>
                            {len < 25 && (
                              <div className="bg-emerald-900/60 flex items-center justify-center text-[9px] text-emerald-300" style={{ width: `${((25 - len) / 25) * 100}%` }}>
                                Pad ({25 - len})
                              </div>
                            )}
                          </div>
                        </div>
                      ))}

                      <div className="text-[10px] text-amber-400 font-mono mt-2">Bucket #2 (Long Sequences: max 140t)</div>
                      {[45, 140].map((len, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-[10px] font-mono">
                          <span className="w-12 text-slate-500">#{idx + 3} ({len}t)</span>
                          <div className="flex-1 flex h-4 rounded overflow-hidden bg-slate-800">
                            <div className="bg-blue-600 flex items-center justify-center text-[9px] text-white" style={{ width: `${(len / 140) * 100}%` }}>
                              Real
                            </div>
                            {len < 140 && (
                              <div className="bg-emerald-900/60 flex items-center justify-center text-[9px] text-emerald-300" style={{ width: `${((140 - len) / 140) * 100}%` }}>
                                Pad ({140 - len})
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: ARCHITECTURAL GUIDE */}
        {activeTab === 'edu' && (
          <div className="p-6 rounded-xl bg-slate-900/60 border border-slate-800/80 space-y-6 text-xs text-slate-300">
            <h3 className="text-base font-semibold text-slate-100 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-amber-400" />
              First-Principles Architectural Foundations
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-slate-850">
                <h4 className="font-semibold text-slate-200 text-sm">1. The Left-Padding Imperative for Decoders</h4>
                <p className="text-slate-400 leading-relaxed">
                  In encoder models (BERT), right-padding is standard. However, for <strong>autoregressive decoder models</strong> (GPT, LLaMA, Mistral),
                  new tokens are appended to the right of each sequence.
                </p>
                <div className="p-3 bg-slate-900 rounded font-mono text-[11px] text-amber-300 space-y-1">
                  <div>Right-pad: [T1, T2, PAD, PAD] (generation appends at index 4, stranding pad tokens)</div>
                  <div>Left-pad:  [PAD, PAD, T1, T2] (all sequences end at index 3, generation appends uniformly at index 4)</div>
                </div>
              </div>

              <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-slate-850">
                <h4 className="font-semibold text-slate-200 text-sm">2. The Quadratic Attention Cost of Padding</h4>
                <p className="text-slate-400 leading-relaxed">
                  Standard self-attention computes Attention = Softmax(Q K^T / sqrt(d_k)), requiring O(L^2) matrix multiplications.
                  When batching sequence lengths up to L_max, the attention compute scales with:
                </p>
                <div className="p-3 bg-slate-900 rounded font-mono text-[11px] text-purple-300">
                  FLOPs_attn = 2 * N * (L_max)^2 * d_model
                  <br />
                  By clustering into length buckets with similar lengths, L_max_bucket &lt;&lt; L_max_global, cutting quadratic waste by 50-80%.
                </div>
              </div>

              <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-slate-850">
                <h4 className="font-semibold text-slate-200 text-sm">3. Asynchronous Priority Worker Pool</h4>
                <p className="text-slate-400 leading-relaxed">
                  Synchronous batch APIs tie up web worker threads for minutes. Libra&apos;s BatchJobScheduler implements an in-memory
                  asynchronous queue decoupled from HTTP requests.
                </p>
                <p className="text-slate-400 leading-relaxed">
                  Clients submit batch jobs via POST /api/v1/batch/jobs, receive a job_id immediately, and stream real-time progress
                  via Server-Sent Events (SSE).
                </p>
              </div>

              <div className="space-y-3 p-4 rounded-xl bg-slate-950 border border-slate-850">
                <h4 className="font-semibold text-slate-200 text-sm">4. CPU Hardware &amp; Zero-Cost Budget</h4>
                <p className="text-slate-400 leading-relaxed">
                  Aligned with Prime Directive #3, the worker pool caps concurrency to 1 thread on Intel i5-12450H CPUs. This prevents
                  context-switching thrashing and guarantees that the interactive chat interface remains responsive during bulk offline inference.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
