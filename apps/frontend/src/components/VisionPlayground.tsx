"use client";

import React, { useState } from "react";
import {
  EmbedImageResponse,
  VLMGenerateResponse,
  embedImage,
  generateWithVLM,
} from "@/lib/api";

export const VisionPlayground: React.FC = () => {
  const [pattern, setPattern] = useState("checkerboard");
  const [prompt, setPrompt] = useState("This image shows a");
  const [embedData, setEmbedData] = useState<EmbedImageResponse | null>(null);
  const [genData, setGenData] = useState<VLMGenerateResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleDecompose = async () => {
    setIsLoading(true);
    try {
      const res = await embedImage(pattern, 32, 8, 32);
      setEmbedData(res);
    } catch (err) {
      console.error("Decomposition failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerate = async () => {
    setIsLoading(true);
    try {
      const res = await generateWithVLM(prompt, pattern, 8);
      setGenData(res);
    } catch (err) {
      console.error("VLM generation failed:", err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 shadow-lg backdrop-blur space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-gray-800 pb-4">
        <div>
          <h3 className="text-base font-semibold text-gray-200 flex items-center gap-2">
            <span className="inline-block w-2.5 h-2.5 rounded-full bg-fuchsia-400 animate-pulse" />
            Vision-Language Multi-Modal Playground
          </h3>
          <p className="text-xs text-gray-400 mt-0.5">
            2D image patch decomposition, linear projection adapter, and autoregressive visual prefix conditioning.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <select
            value={pattern}
            onChange={(e) => setPattern(e.target.value)}
            className="bg-gray-950 border border-gray-800 rounded-lg px-2.5 py-1 text-xs text-gray-200 focus:outline-none focus:border-fuchsia-500"
          >
            <option value="checkerboard">Checkerboard Pattern</option>
            <option value="horizontal_gradient">Luminance Gradient</option>
            <option value="solid">Solid Tone</option>
          </select>

          <button
            onClick={handleDecompose}
            disabled={isLoading}
            className="px-3 py-1.5 rounded-lg text-xs font-medium bg-gray-800 hover:bg-gray-700 text-gray-200 border border-gray-700"
          >
            {isLoading ? "Decomposing..." : "Decompose Patches"}
          </button>

          <button
            onClick={handleGenerate}
            disabled={isLoading}
            className="px-3.5 py-1.5 rounded-lg text-xs font-medium bg-fuchsia-600 hover:bg-fuchsia-500 text-white shadow-sm"
          >
            {isLoading ? "Generating..." : "Generate Description"}
          </button>
        </div>
      </div>

      {embedData && (
        <div className="space-y-3">
          <div className="flex items-center justify-between text-xs text-gray-400 bg-gray-950/40 p-2.5 rounded border border-gray-800">
            <span>Image Size: <strong className="text-gray-200">{embedData.image_size}x{embedData.image_size}</strong></span>
            <span>Patch Size: <strong className="text-gray-200">{embedData.patch_size}x{embedData.patch_size}</strong></span>
            <span>Visual Tokens: <strong className="text-fuchsia-400">{embedData.num_patches} patches</strong></span>
            <span>Projected Dimension: <strong className="text-gray-200">{embedData.vision_dim} &rarr; {embedData.projected_llm_dim}</strong></span>
          </div>

          <div className="grid grid-cols-4 gap-2 max-w-xs mx-auto p-3 bg-gray-950 rounded-lg border border-gray-800">
            {embedData.patch_grid.map((patch) => {
              const [r, g, b] = patch.mean_rgb;
              const bgStyle = `rgb(${Math.round(r * 255)}, ${Math.round(g * 255)}, ${Math.round(b * 255)})`;
              return (
                <div
                  key={patch.patch_index}
                  style={{ backgroundColor: bgStyle }}
                  className="w-12 h-12 rounded border border-gray-700/60 flex items-center justify-center text-[10px] font-mono text-white shadow-inner"
                  title={`Patch #${patch.patch_index} (Norm: ${patch.norm})`}
                >
                  #{patch.patch_index}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {genData && (
        <div className="rounded-lg bg-gray-950/70 p-3.5 border border-fuchsia-900/30 text-xs space-y-1.5">
          <div className="flex items-center justify-between text-gray-400">
            <span className="font-semibold text-gray-200">Model Multi-Modal Description</span>
            <span className="font-mono text-gray-500">{genData.latency_ms} ms</span>
          </div>
          <div className="text-fuchsia-300 font-medium text-sm">
            {genData.generated_text}
          </div>
          <div className="text-gray-500 text-[11px]">
            Visual prefix tokens: {genData.num_visual_tokens} | Generated token IDs: {genData.generated_tokens.join(", ")}
          </div>
        </div>
      )}
    </div>
  );
};
