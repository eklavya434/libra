"use client";

import React, { useState } from "react";
import {
  NeedleEvaluationResponse,
  NeedleResultItem,
  runNeedleEvaluation,
} from "@/lib/api";

interface LongContextHeatmapProps {
  model?: string;
}

export const LongContextHeatmap: React.FC<LongContextHeatmapProps> = ({
  model = "mock",
}) => {
  const [data, setData] = useState<NeedleEvaluationResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCell, setSelectedCell] = useState<NeedleResultItem | null>(null);

  const contextLengths = [250, 500, 1000, 2000];
  const depthPercentages = [0.0, 25.0, 50.0, 75.0, 100.0];

  const handleRunEvaluation = async () => {
    setIsLoading(true);
    setSelectedCell(null);
    try {
      const res = await runNeedleEvaluation(
        model,
        contextLengths,
        depthPercentages.map((d) => d / 100.0)
      );
      setData(res);
    } catch (err) {
      console.error("Failed to run needle evaluation:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const findCell = (len: number, depth: number): NeedleResultItem | undefined => {
    if (!data) return undefined;
    return data.results.find(
      (r) => r.context_length === len && Math.abs(r.depth_percent - depth) < 1.0
    );
  };

  const getCellColor = (item?: NeedleResultItem) => {
    if (!item) return "bg-gray-800/40 border-gray-700/50 text-gray-500";
    if (item.score >= 0.99) {
      return "bg-emerald-500/20 border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/30";
    }
    if (item.score > 0.0) {
      return "bg-amber-500/20 border-amber-500/50 text-amber-400 hover:bg-amber-500/30";
    }
    return "bg-rose-500/20 border-rose-500/50 text-rose-400 hover:bg-rose-500/30";
  };

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 shadow-lg backdrop-blur">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-gray-800 pb-4">
        <div>
          <h3 className="text-base font-semibold text-gray-200 flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-indigo-500 animate-pulse" />
            Needle-In-A-Haystack Retrieval Heatmap
          </h3>
          <p className="text-xs text-gray-400 mt-0.5">
            Evaluates factual associative recall across sequence context lengths and document insertion depths.
          </p>
        </div>

        <button
          onClick={handleRunEvaluation}
          disabled={isLoading}
          className={`px-3.5 py-1.5 rounded-lg text-xs font-medium transition-all ${
            isLoading
              ? "bg-gray-800 text-gray-500 cursor-not-allowed"
              : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm hover:shadow-indigo-500/20"
          }`}
        >
          {isLoading ? "Running Matrix..." : "Run Evaluation Matrix"}
        </button>
      </div>

      {data && (
        <div className="grid grid-cols-3 gap-3 my-4">
          <div className="rounded-lg bg-gray-950/60 p-2.5 border border-gray-800/80">
            <div className="text-[11px] text-gray-400 uppercase font-medium">Retrieval Accuracy</div>
            <div className="text-base font-semibold text-emerald-400 mt-0.5">
              {data.accuracy_percent}%
            </div>
          </div>
          <div className="rounded-lg bg-gray-950/60 p-2.5 border border-gray-800/80">
            <div className="text-[11px] text-gray-400 uppercase font-medium">Total Evaluations</div>
            <div className="text-base font-semibold text-gray-200 mt-0.5">
              {data.total_trials}
            </div>
          </div>
          <div className="rounded-lg bg-gray-950/60 p-2.5 border border-gray-800/80">
            <div className="text-[11px] text-gray-400 uppercase font-medium">Average Latency</div>
            <div className="text-base font-semibold text-indigo-400 mt-0.5">
              {data.average_latency_ms} ms
            </div>
          </div>
        </div>
      )}

      {/* Heatmap Grid */}
      <div className="mt-4 overflow-x-auto">
        <div className="inline-block min-w-full align-middle">
          <table className="min-w-full text-center border-separate border-spacing-1.5">
            <thead>
              <tr>
                <th className="text-[11px] font-medium text-gray-500 uppercase px-2 py-1">
                  Depth \ Length
                </th>
                {contextLengths.map((len) => (
                  <th
                    key={len}
                    className="text-[11px] font-medium text-gray-300 uppercase px-3 py-1 bg-gray-950/40 rounded border border-gray-800/60"
                  >
                    {len} tokens
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {depthPercentages.map((depth) => (
                <tr key={depth}>
                  <td className="text-[11px] font-medium text-gray-400 px-2 py-1 text-right whitespace-nowrap bg-gray-950/40 rounded border border-gray-800/60">
                    {depth}%
                  </td>
                  {contextLengths.map((len) => {
                    const item = findCell(len, depth);
                    const isSelected =
                      selectedCell &&
                      selectedCell.context_length === len &&
                      Math.abs(selectedCell.depth_percent - depth) < 1.0;
                    return (
                      <td
                        key={`${len}-${depth}`}
                        onClick={() => item && setSelectedCell(item)}
                        className={`cursor-pointer rounded border p-2.5 text-xs font-mono transition-all ${getCellColor(
                          item
                        )} ${isSelected ? "ring-2 ring-indigo-400 scale-105" : ""}`}
                        title={
                          item
                            ? `Depth: ${item.depth_percent}%, Len: ${item.context_length}, Score: ${item.score}`
                            : "Click Run Evaluation to benchmark"
                        }
                      >
                        {item ? (item.score >= 0.99 ? "100%" : `${Math.round(item.score * 100)}%`) : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {selectedCell && (
        <div className="mt-4 rounded-lg border border-gray-800 bg-gray-950/70 p-3.5 text-xs">
          <div className="flex items-center justify-between text-gray-400 mb-1.5">
            <span className="font-semibold text-gray-200">
              Trial Inspection ({selectedCell.context_length} tokens @ {selectedCell.depth_percent}% depth)
            </span>
            <span className="font-mono text-gray-500">{selectedCell.latency_ms} ms</span>
          </div>
          <div className="text-gray-300">
            <span className="text-gray-500 font-medium">Model Output: </span>
            {selectedCell.retrieved_text || "<Empty Response>"}
          </div>
        </div>
      )}
    </div>
  );
};
