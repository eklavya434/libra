"use client";

import React, { useState } from "react";
import {
  BatchRunResponse,
  PagedSimulationResult,
  runContinuousBatchSimulation,
  simulatePagedMemory,
} from "@/lib/api";

export const PagedMemoryInspector: React.FC = () => {
  const [simResult, setSimResult] = useState<PagedSimulationResult | null>(null);
  const [batchResult, setBatchResult] = useState<BatchRunResponse | null>(null);
  const [isLoadingSim, setIsLoadingSim] = useState(false);
  const [isLoadingBatch, setIsLoadingBatch] = useState(false);

  const handleSimulate = async () => {
    setIsLoadingSim(true);
    try {
      const res = await simulatePagedMemory(8, 120, 256, 16);
      setSimResult(res);
    } catch (err) {
      console.error("Simulation failed:", err);
    } finally {
      setIsLoadingSim(false);
    }
  };

  const handleRunBatch = async () => {
    setIsLoadingBatch(true);
    try {
      const res = await runContinuousBatchSimulation();
      setBatchResult(res);
    } catch (err) {
      console.error("Batch run failed:", err);
    } finally {
      setIsLoadingBatch(false);
    }
  };

  const formatMB = (bytes: number) => (bytes / (1024 * 1024)).toFixed(2);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 shadow-lg backdrop-blur space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-gray-800 pb-4">
        <div>
          <h3 className="text-base font-semibold text-gray-200 flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
            PagedAttention & Continuous Batching Inspector
          </h3>
          <p className="text-xs text-gray-400 mt-0.5">
            Virtual memory block page management for KV caches eliminating internal/external memory fragmentation.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleSimulate}
            disabled={isLoadingSim}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700"
          >
            {isLoadingSim ? "Auditing..." : "Audit Memory Savings"}
          </button>
          <button
            onClick={handleRunBatch}
            disabled={isLoadingBatch}
            className="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-cyan-600 hover:bg-cyan-500 text-white shadow-sm"
          >
            {isLoadingBatch ? "Simulating..." : "Run Continuous Batch"}
          </button>
        </div>
      </div>

      {simResult && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="rounded-lg bg-gray-950/60 p-3 border border-rose-900/30">
            <div className="text-[11px] text-gray-400 uppercase font-medium">Contiguous Cache Waste</div>
            <div className="text-lg font-semibold text-rose-400 mt-0.5">
              {simResult.contiguous_waste_percent}%
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {formatMB(simResult.contiguous_reservation_bytes)} MB reserved upfront
            </div>
          </div>

          <div className="rounded-lg bg-gray-950/60 p-3 border border-cyan-900/30">
            <div className="text-[11px] text-gray-400 uppercase font-medium">PagedAttention Fragmentation</div>
            <div className="text-lg font-semibold text-cyan-400 mt-0.5">
              {simResult.paged_internal_frag_percent}%
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {formatMB(simResult.paged_allocation_bytes)} MB dynamically allocated
            </div>
          </div>

          <div className="rounded-lg bg-gray-950/60 p-3 border border-emerald-900/30">
            <div className="text-[11px] text-gray-400 uppercase font-medium">Concurrency Multiplier</div>
            <div className="text-lg font-semibold text-emerald-400 mt-0.5">
              {simResult.max_concurrency_multiplier}x
            </div>
            <div className="text-xs text-emerald-500/80 mt-1">
              {simResult.memory_savings_percent}% memory reduction
            </div>
          </div>
        </div>
      )}

      {batchResult && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-gray-400 bg-gray-950/40 p-2.5 rounded border border-gray-800">
            <span>Throughput: <strong className="text-cyan-400">{batchResult.throughput_tokens_per_sec} tok/s</strong></span>
            <span>Total Tokens: <strong className="text-gray-200">{batchResult.total_tokens_generated}</strong></span>
            <span>Iterations: <strong className="text-gray-200">{batchResult.total_iterations}</strong></span>
            <span>Duration: <strong className="text-gray-200">{batchResult.total_time_ms} ms</strong></span>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-xs text-left text-gray-400">
              <thead className="bg-gray-950/60 uppercase text-[10px] text-gray-500">
                <tr>
                  <th className="px-2 py-1.5">Iter</th>
                  <th className="px-2 py-1.5">Running</th>
                  <th className="px-2 py-1.5">Waiting</th>
                  <th className="px-2 py-1.5">Finished</th>
                  <th className="px-2 py-1.5">Allocated Blocks</th>
                  <th className="px-2 py-1.5">Free Blocks</th>
                  <th className="px-2 py-1.5">Step Time</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/60 font-mono">
                {batchResult.timeline.slice(0, 10).map((r) => (
                  <tr key={r.iteration} className="hover:bg-gray-800/20">
                    <td className="px-2 py-1">{r.iteration}</td>
                    <td className="px-2 py-1 text-emerald-400">{r.running}</td>
                    <td className="px-2 py-1 text-amber-400">{r.waiting}</td>
                    <td className="px-2 py-1 text-gray-400">{r.finished}</td>
                    <td className="px-2 py-1 text-cyan-400">{r.allocated_blocks}</td>
                    <td className="px-2 py-1 text-gray-400">{r.free_blocks}</td>
                    <td className="px-2 py-1">{r.elapsed_ms} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
