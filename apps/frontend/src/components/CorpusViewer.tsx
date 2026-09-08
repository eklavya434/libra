"use client";

import React, { useState } from "react";
import {
  FormatChatResponse,
  PackResponse,
  formatChatML,
  packCorpusSequences,
} from "@/lib/api";

export const CorpusViewer: React.FC = () => {
  const [chatData, setChatData] = useState<FormatChatResponse | null>(null);
  const [packData, setPackData] = useState<PackResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const sampleMessages = [
    { role: "system", content: "You are Libra, an educational AI assistant." },
    { role: "user", content: "Explain gradient descent in one sentence." },
    {
      role: "assistant",
      content:
        "Gradient descent iteratively updates parameters in the direction of steepest descent to minimize loss.",
    },
  ];

  const handleInspectChatML = async () => {
    setIsLoading(true);
    try {
      const res = await formatChatML(sampleMessages, 256);
      setChatData(res);
    } catch (err) {
      console.error("Failed to format ChatML:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handlePackCorpus = async () => {
    setIsLoading(true);
    try {
      const res = await packCorpusSequences(256);
      setPackData(res);
    } catch (err) {
      console.error("Failed to pack sequences:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 shadow-lg backdrop-blur space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-gray-800 pb-4">
        <div>
          <h3 className="text-base font-semibold text-gray-200 flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            Instruction Corpus & Loss Masking Inspector
          </h3>
          <p className="text-xs text-gray-400 mt-0.5">
            ChatML template formatting, completion-only prompt loss masking (-100), and sequence packing.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleInspectChatML}
            disabled={isLoading}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700"
          >
            {isLoading ? "Formatting..." : "Inspect ChatML Masking"}
          </button>
          <button
            onClick={handlePackCorpus}
            disabled={isLoading}
            className="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm"
          >
            {isLoading ? "Packing..." : "Simulate Multi-Turn Packing"}
          </button>
        </div>
      </div>

      {chatData && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-3 text-xs">
            <div className="bg-gray-950/60 p-2.5 rounded border border-gray-800">
              <div className="text-gray-500 text-[10px] uppercase">Total Tokens</div>
              <div className="text-base font-semibold text-gray-200">{chatData.total_tokens}</div>
            </div>
            <div className="bg-gray-950/60 p-2.5 rounded border border-gray-800">
              <div className="text-gray-500 text-[10px] uppercase">Trainable Tokens</div>
              <div className="text-base font-semibold text-emerald-400">{chatData.trainable_tokens}</div>
            </div>
            <div className="bg-gray-950/60 p-2.5 rounded border border-gray-800">
              <div className="text-gray-500 text-[10px] uppercase">Masked Tokens</div>
              <div className="text-base font-semibold text-gray-400">{chatData.masked_tokens}</div>
            </div>
            <div className="bg-gray-950/60 p-2.5 rounded border border-gray-800">
              <div className="text-gray-500 text-[10px] uppercase">Trainable Fraction</div>
              <div className="text-base font-semibold text-indigo-400">
                {(chatData.trainable_fraction * 100).toFixed(1)}%
              </div>
            </div>
          </div>

          <div className="rounded-lg bg-gray-950 p-3.5 border border-gray-800 font-mono text-xs leading-relaxed max-h-48 overflow-y-auto">
            <div className="text-[11px] text-gray-500 mb-2 font-sans">
              <span className="inline-block w-2.5 h-2.5 rounded bg-emerald-500/30 border border-emerald-500/80 mr-1.5 align-middle" />
              Trainable Assistant Tokens (Active Gradient)
              <span className="inline-block w-2.5 h-2.5 rounded bg-gray-800 border border-gray-700 ml-4 mr-1.5 align-middle" />
              Masked Prompt Tokens (-100, Ignored)
            </div>
            {chatData.tokens.map((t) => (
              <span
                key={t.token_idx}
                className={`inline-block px-1 py-0.5 m-0.5 rounded text-[11px] ${
                  t.is_masked
                    ? "bg-gray-800/80 text-gray-400 border border-gray-700/50"
                    : "bg-emerald-500/20 text-emerald-300 border border-emerald-500/50 font-semibold"
                }`}
                title={`Token #${t.token_idx} (ID: ${t.token_id}) | Label: ${t.label}`}
              >
                {t.char_repr === " " ? "␣" : t.char_repr}
              </span>
            ))}
          </div>
        </div>
      )}

      {packData && (
        <div className="rounded-lg bg-gray-950/60 p-4 border border-emerald-900/30 space-y-3">
          <div className="flex items-center justify-between text-xs text-gray-300">
            <span>Original Dialogues: <strong className="text-gray-100">{packData.original_dialogues}</strong></span>
            <span>Packed Sequences: <strong className="text-emerald-400">{packData.packed_sequences}</strong></span>
            <span>Efficiency Gain: <strong className="text-emerald-400">+{packData.efficiency_gain_percent}%</strong></span>
          </div>

          <div className="text-xs text-gray-400">
            Unpacked padded tokens: <strong className="text-rose-400">{packData.tokens_unpacked_with_padding}</strong> &rarr; Packed tokens: <strong className="text-emerald-400">{packData.tokens_packed}</strong>
          </div>
        </div>
      )}
    </div>
  );
};

export default CorpusViewer;
