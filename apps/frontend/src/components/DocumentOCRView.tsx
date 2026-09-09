'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  FileSearch,
  Table,
  Sigma,
  Key,
  Layers,
  Sparkles,
  Send,
  Eye,
  CheckCircle2,
  BookOpen,
  Maximize2,
  Minimize2,
  RotateCcw,
} from 'lucide-react';

interface BoundingBoxData {
  x_min: number;
  y_min: number;
  x_max: number;
  y_max: number;
}

interface DocumentElementData {
  element_id: string;
  type: 'heading' | 'paragraph' | 'table' | 'equation' | 'key_value' | 'list_item';
  content: string;
  bbox: BoundingBoxData;
  page_number: number;
  reading_order: number;
  confidence: number;
  table_data?: {
    headers: string[];
    rows: string[][];
    num_rows: number;
    num_cols: number;
  };
  latex_formula?: string;
  metadata?: Record<string, any>;
}

interface DocumentPageData {
  page_number: number;
  width: number;
  height: number;
  num_elements: number;
  elements: DocumentElementData[];
}

interface ParsedDocumentData {
  doc_id: string;
  title: string;
  num_pages: number;
  total_elements: number;
  pages: DocumentPageData[];
}

interface DocumentCitationData {
  page_number: number;
  element_id: string;
  element_type: string;
  snippet: string;
  bbox: BoundingBoxData;
  cell_reference?: string | null;
  confidence: number;
}

interface GroundedQAResultData {
  query: string;
  answer: string;
  citations: DocumentCitationData[];
  relevant_element_ids: string[];
}

interface DocumentPreset {
  id: string;
  title: string;
  description: string;
  text: string;
}

export default function DocumentOCRView() {
  const [presets, setPresets] = useState<DocumentPreset[]>([]);
  const [selectedPresetId, setSelectedPresetId] = useState<string>('financial_quarterly_report');
  const [customText, setCustomText] = useState<string>('');
  const [isCustomMode, setIsCustomMode] = useState<boolean>(false);

  const [documentData, setDocumentData] = useState<ParsedDocumentData | null>(null);
  const [activeElementId, setActiveElementId] = useState<string | null>(null);
  const [highlightedElementIds, setHighlightedElementIds] = useState<string[]>([]);

  // Right Panel Mode
  const [rightTab, setRightTab] = useState<'qa' | 'ast' | 'edu'>('qa');

  // Grounded Q&A state
  const [question, setQuestion] = useState<string>('What was the Total Cloud Revenue in Q3 2026?');
  const [qaResult, setQaResult] = useState<GroundedQAResultData | null>(null);
  const [loadingQA, setLoadingQA] = useState<boolean>(false);
  const [loadingDoc, setLoadingDoc] = useState<boolean>(false);

  // Load presets on mount
  useEffect(() => {
    async function loadPresets() {
      try {
        const res = await fetch('/api/v1/document/presets');
        if (res.ok) {
          const data = await res.json();
          setPresets(data.presets || []);
          if (data.presets && data.presets.length > 0) {
            parsePresetText(data.presets[0].text, data.presets[0].title);
          }
        }
      } catch (err) {
        console.error('Failed to load document presets:', err);
      }
    }
    loadPresets();
  }, []);

  const parsePresetText = async (text: string, title?: string) => {
    try {
      setLoadingDoc(true);
      const res = await fetch('/api/v1/document/parse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, title }),
      });
      if (res.ok) {
        const data = await res.json();
        setDocumentData(data.document);
        setActiveElementId(null);
        setHighlightedElementIds([]);
        setQaResult(null);
      }
    } catch (err) {
      console.error('Failed to parse document:', err);
    } finally {
      setLoadingDoc(false);
    }
  };

  const handleSelectPreset = (preset: DocumentPreset) => {
    setSelectedPresetId(preset.id);
    setIsCustomMode(false);
    parsePresetText(preset.text, preset.title);
    if (preset.id === 'financial_quarterly_report') {
      setQuestion('What was the Total Cloud Revenue in Q3 2026?');
    } else if (preset.id === 'transformer_research_paper') {
      setQuestion('What is the formula for multi-head attention?');
    } else {
      setQuestion('What is the Grand Total Due on this invoice?');
    }
  };

  const handleAskQuestion = async (qText?: string) => {
    const targetQ = qText || question;
    if (!targetQ.trim() || !documentData) return;

    try {
      setLoadingQA(true);
      const activeText = isCustomMode
        ? customText
        : presets.find((p) => p.id === selectedPresetId)?.text || '';

      const res = await fetch('/api/v1/document/qa', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: activeText, query: targetQ }),
      });
      if (res.ok) {
        const data = await res.json();
        const result: GroundedQAResultData = data.result;
        setQaResult(result);
        setHighlightedElementIds(result.relevant_element_ids || []);
        if (result.relevant_element_ids && result.relevant_element_ids.length > 0) {
          setActiveElementId(result.relevant_element_ids[0]);
        }
      }
    } catch (err) {
      console.error('Failed to perform grounded QA:', err);
    } finally {
      setLoadingQA(false);
    }
  };

  const getElementColor = (type: string, isHighlighted: boolean, isActive: boolean) => {
    if (isActive) return 'border-amber-400 bg-amber-500/20 ring-2 ring-amber-400/80 shadow-lg shadow-amber-500/30';
    if (isHighlighted) return 'border-emerald-400 bg-emerald-500/20 ring-2 ring-emerald-400/60 shadow-md';
    switch (type) {
      case 'heading':
        return 'border-blue-500/60 bg-blue-500/10 hover:border-blue-400 hover:bg-blue-500/20';
      case 'table':
        return 'border-emerald-500/60 bg-emerald-500/10 hover:border-emerald-400 hover:bg-emerald-500/20';
      case 'equation':
        return 'border-purple-500/60 bg-purple-500/10 hover:border-purple-400 hover:bg-purple-500/20';
      case 'key_value':
        return 'border-amber-500/60 bg-amber-500/10 hover:border-amber-400 hover:bg-amber-500/20';
      default:
        return 'border-slate-700/60 bg-slate-900/40 hover:border-slate-500 hover:bg-slate-800/40';
    }
  };

  const activeElement = documentData?.pages[0]?.elements.find((e) => e.element_id === activeElementId);

  return (
    <div className="flex flex-col h-full bg-slate-950 text-slate-100 overflow-hidden">
      {/* Top Header */}
      <div className="p-4 border-b border-slate-800/80 bg-slate-900/40 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-lg shadow-cyan-500/20">
            <FileSearch className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-bold text-lg text-slate-100">Multi-Modal Document Understanding (LibraOCR)</h2>
              <span className="text-[11px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-mono">
                Phase 40
              </span>
            </div>
            <p className="text-xs text-slate-400">
              Visual layout AST, spatial 2D bounding boxes, markdown table matrices, and grounded citations.
            </p>
          </div>
        </div>

        {/* Preset Selector */}
        <div className="flex flex-wrap items-center gap-2">
          {presets.map((p) => (
            <button
              key={p.id}
              onClick={() => handleSelectPreset(p)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                selectedPresetId === p.id && !isCustomMode
                  ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                  : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              {p.id === 'financial_quarterly_report' && 'Financial Report'}
              {p.id === 'transformer_research_paper' && 'Research Paper'}
              {p.id === 'commercial_tax_invoice' && 'Service Invoice'}
            </button>
          ))}
          <button
            onClick={() => setIsCustomMode(true)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
              isCustomMode
                ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/30'
                : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
            }`}
          >
            Custom Text
          </button>
        </div>
      </div>

      {/* Main Split Body */}
      <div className="flex-1 overflow-hidden grid grid-cols-1 lg:grid-cols-12 gap-0">
        {/* LEFT COLUMN: Visual Document Page Canvas (7 cols) */}
        <div className="lg:col-span-7 h-full flex flex-col border-r border-slate-800/80 bg-slate-950 p-4 overflow-y-auto">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-xs font-semibold text-slate-300">
              <Eye className="w-4 h-4 text-cyan-400" />
              <span>Visual Document Canvas (Page 1 of {documentData?.num_pages || 1})</span>
            </div>
            <div className="flex items-center gap-3 text-[11px] font-mono text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-blue-500 inline-block" />
                Heading
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-emerald-500 inline-block" />
                Table
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-purple-500 inline-block" />
                Equation
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block" />
                Key-Value
              </span>
            </div>
          </div>

          {/* Custom Input Editor Mode */}
          {isCustomMode ? (
            <div className="flex-1 flex flex-col space-y-3">
              <textarea
                value={customText}
                onChange={(e) => setCustomText(e.target.value)}
                placeholder="Type or paste markdown document text with tables (| Col A | Col B |), formulas ($$ ... $$), and key-values..."
                className="flex-1 w-full bg-slate-900 border border-slate-800 rounded-xl p-4 text-xs font-mono text-slate-200 resize-none focus:outline-none focus:border-cyan-500"
              />
              <button
                onClick={() => parsePresetText(customText, 'Custom Document')}
                className="bg-cyan-600 hover:bg-cyan-500 text-white font-medium py-2 px-4 rounded-lg text-xs self-end"
              >
                Parse Document Layout
              </button>
            </div>
          ) : (
            /* 2D Document Page Simulation Canvas */
            <div className="flex-1 relative rounded-2xl bg-slate-900/70 border border-slate-800 shadow-2xl p-4 overflow-y-auto min-h-[620px]">
              {loadingDoc ? (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-950/60 z-20">
                  <RotateCcw className="w-6 h-6 animate-spin text-cyan-400" />
                </div>
              ) : null}

              {/* Elements positioned by bounding box coordinates */}
              {documentData?.pages[0]?.elements.map((elem) => {
                const isHighlighted = highlightedElementIds.includes(elem.element_id);
                const isActive = activeElementId === elem.element_id;
                const colorClasses = getElementColor(elem.type, isHighlighted, isActive);

                return (
                  <div
                    key={elem.element_id}
                    onClick={() => setActiveElementId(elem.element_id)}
                    className={`my-3 p-3.5 rounded-xl border transition-all cursor-pointer relative group ${colorClasses}`}
                  >
                    {/* Element Type Badge & Spatial Coords Header */}
                    <div className="flex items-center justify-between mb-1.5 text-[10px] font-mono text-slate-400">
                      <div className="flex items-center gap-1.5">
                        {elem.type === 'heading' && <span className="text-blue-400 font-bold uppercase">HEADING</span>}
                        {elem.type === 'table' && <span className="text-emerald-400 font-bold uppercase">TABLE MATRIX</span>}
                        {elem.type === 'equation' && <span className="text-purple-400 font-bold uppercase">LATEX EQUATION</span>}
                        {elem.type === 'key_value' && <span className="text-amber-400 font-bold uppercase">KEY-VALUE</span>}
                        {elem.type === 'paragraph' && <span className="text-slate-400 font-bold uppercase">TEXT BLOCK</span>}
                        <span className="text-slate-500">({elem.element_id})</span>
                      </div>
                      <span className="text-[9px] text-slate-500">
                        [{elem.bbox.x_min.toFixed(2)}, {elem.bbox.y_min.toFixed(2)}, {elem.bbox.x_max.toFixed(2)}, {elem.bbox.y_max.toFixed(2)}]
                      </span>
                    </div>

                    {/* Element Content Rendering */}
                    {elem.type === 'table' && elem.table_data ? (
                      <div className="overflow-x-auto my-1">
                        <table className="min-w-full text-left text-xs font-mono border-collapse border border-slate-800">
                          <thead>
                            <tr className="bg-slate-800/60">
                              {elem.table_data.headers.map((h, i) => (
                                <th key={i} className="border border-slate-700/80 px-2 py-1 text-slate-200 font-semibold">
                                  {h}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {elem.table_data.rows.map((row, r_idx) => (
                              <tr key={r_idx} className="hover:bg-slate-800/40">
                                {row.map((cell, c_idx) => (
                                  <td key={c_idx} className="border border-slate-800/80 px-2 py-1 text-slate-300">
                                    {cell}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : elem.type === 'equation' ? (
                      <div className="my-2 p-2.5 rounded-lg bg-slate-950/80 font-mono text-xs text-purple-300 border border-purple-900/40 text-center">
                        {elem.content}
                      </div>
                    ) : elem.type === 'heading' ? (
                      <div className="font-bold text-sm text-slate-100">{elem.content}</div>
                    ) : elem.type === 'key_value' ? (
                      <div className="text-xs font-mono text-amber-200">{elem.content}</div>
                    ) : (
                      <div className="text-xs text-slate-300 leading-relaxed">{elem.content}</div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* RIGHT COLUMN: Grounded Q&A & Inspection Drawer (5 cols) */}
        <div className="lg:col-span-5 h-full flex flex-col bg-slate-900/40 p-4 overflow-y-auto space-y-4">
          {/* Sub Tab Switcher */}
          <div className="flex items-center bg-slate-900 p-1 rounded-lg border border-slate-800 text-xs">
            <button
              onClick={() => setRightTab('qa')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md font-medium transition-all ${
                rightTab === 'qa' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Grounded Q&amp;A</span>
            </button>
            <button
              onClick={() => setRightTab('ast')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md font-medium transition-all ${
                rightTab === 'ast' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Layers className="w-3.5 h-3.5" />
              <span>Element Inspector</span>
            </button>
            <button
              onClick={() => setRightTab('edu')}
              className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md font-medium transition-all ${
                rightTab === 'edu' ? 'bg-cyan-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>Guide</span>
            </button>
          </div>

          {/* TAB 1: Grounded Document Q&A */}
          {rightTab === 'qa' && (
            <div className="space-y-4 flex-1 flex flex-col">
              <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-3">
                <span className="text-xs font-semibold text-slate-200 block">Ask Grounded Question</span>

                {/* Question Input */}
                <div className="relative">
                  <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAskQuestion()}
                    placeholder="Ask about tables, totals, formulas, or numbers..."
                    className="w-full bg-slate-950 border border-slate-800 rounded-lg pl-3 pr-10 py-2.5 text-xs text-slate-200 focus:outline-none focus:border-cyan-500"
                  />
                  <button
                    onClick={() => handleAskQuestion()}
                    disabled={loadingQA}
                    className="absolute right-2 top-2 p-1 text-cyan-400 hover:text-cyan-300 disabled:opacity-50"
                  >
                    {loadingQA ? <RotateCcw className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                  </button>
                </div>

                {/* Preset Prompt Suggestions */}
                <div className="space-y-1.5 pt-1">
                  <span className="text-[10px] text-slate-400 block uppercase tracking-wider font-mono">
                    Suggested Questions
                  </span>
                  {selectedPresetId === 'financial_quarterly_report' && (
                    <>
                      <button
                        onClick={() => {
                          setQuestion('What was the Total Cloud Revenue in Q3 2026?');
                          handleAskQuestion('What was the Total Cloud Revenue in Q3 2026?');
                        }}
                        className="text-left w-full text-[11px] text-cyan-400 hover:text-cyan-300 truncate"
                      >
                        • What was the Total Cloud Revenue in Q3 2026?
                      </button>
                      <button
                        onClick={() => {
                          setQuestion('What was the Operating Income and YoY growth?');
                          handleAskQuestion('What was the Operating Income and YoY growth?');
                        }}
                        className="text-left w-full text-[11px] text-cyan-400 hover:text-cyan-300 truncate"
                      >
                        • What was the Operating Income and YoY growth?
                      </button>
                      <button
                        onClick={() => {
                          setQuestion('What is the formula for operating efficiency?');
                          handleAskQuestion('What is the formula for operating efficiency?');
                        }}
                        className="text-left w-full text-[11px] text-cyan-400 hover:text-cyan-300 truncate"
                      >
                        • What is the formula for operating efficiency?
                      </button>
                    </>
                  )}
                  {selectedPresetId === 'transformer_research_paper' && (
                    <>
                      <button
                        onClick={() => {
                          setQuestion('What is the formula for multi-head attention?');
                          handleAskQuestion('What is the formula for multi-head attention?');
                        }}
                        className="text-left w-full text-[11px] text-cyan-400 hover:text-cyan-300 truncate"
                      >
                        • What is the formula for multi-head attention?
                      </button>
                      <button
                        onClick={() => {
                          setQuestion('How many layers and heads does the Large tier have?');
                          handleAskQuestion('How many layers and heads does the Large tier have?');
                        }}
                        className="text-left w-full text-[11px] text-cyan-400 hover:text-cyan-300 truncate"
                      >
                        • How many layers and heads does the Large tier have?
                      </button>
                    </>
                  )}
                  {selectedPresetId === 'commercial_tax_invoice' && (
                    <>
                      <button
                        onClick={() => {
                          setQuestion('What is the Invoice Number and Due Date?');
                          handleAskQuestion('What is the Invoice Number and Due Date?');
                        }}
                        className="text-left w-full text-[11px] text-cyan-400 hover:text-cyan-300 truncate"
                      >
                        • What is the Invoice Number and Due Date?
                      </button>
                      <button
                        onClick={() => {
                          setQuestion('What is the Grand Total Due?');
                          handleAskQuestion('What is the Grand Total Due?');
                        }}
                        className="text-left w-full text-[11px] text-cyan-400 hover:text-cyan-300 truncate"
                      >
                        • What is the Grand Total Due?
                      </button>
                    </>
                  )}
                </div>
              </div>

              {/* Synthesized Answer Card */}
              {qaResult ? (
                <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-3">
                  <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>Grounded Answer</span>
                  </div>
                  <p className="text-xs text-slate-200 leading-relaxed">{qaResult.answer}</p>

                  {/* Citations List */}
                  <div className="space-y-2 pt-2 border-t border-slate-800">
                    <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block">
                      Visual Citations ({qaResult.citations.length})
                    </span>
                    <div className="space-y-2 max-h-48 overflow-y-auto pr-1">
                      {qaResult.citations.map((cit, idx) => (
                        <div
                          key={idx}
                          onClick={() => setActiveElementId(cit.element_id)}
                          className="p-2.5 rounded-lg bg-slate-950 border border-emerald-900/50 hover:border-emerald-500/80 cursor-pointer transition-all text-xs space-y-1"
                        >
                          <div className="flex justify-between items-center text-[10px] font-mono text-emerald-400">
                            <span>
                              Page {cit.page_number} • {cit.element_type.toUpperCase()}
                            </span>
                            <span>Confidence: {Math.round(cit.confidence * 100)}%</span>
                          </div>
                          {cit.cell_reference && (
                            <div className="text-[11px] font-mono text-slate-300 bg-slate-900 p-1.5 rounded border border-slate-800">
                              {cit.cell_reference}
                            </div>
                          )}
                          <div className="text-[10px] font-mono text-slate-500">
                            BBox: [{cit.bbox.x_min.toFixed(2)}, {cit.bbox.y_min.toFixed(2)}, {cit.bbox.x_max.toFixed(2)}, {cit.bbox.y_max.toFixed(2)}]
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="p-6 rounded-xl bg-slate-900/30 border border-slate-800/60 text-center text-xs text-slate-500">
                  Ask a question to view grounded visual citations anchored to table cells and bounding boxes.
                </div>
              )}
            </div>
          )}

          {/* TAB 2: Element Inspector */}
          {rightTab === 'ast' && (
            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-3 flex-1">
              <span className="text-xs font-semibold text-slate-200 block">Selected Layout AST Node</span>
              {activeElement ? (
                <div className="space-y-2.5 text-xs">
                  <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                    <div className="p-2 rounded bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-500">ID</div>
                      <div className="text-slate-200 font-bold">{activeElement.element_id}</div>
                    </div>
                    <div className="p-2 rounded bg-slate-950 border border-slate-800">
                      <div className="text-[10px] text-slate-500">TYPE</div>
                      <div className="text-cyan-400 font-bold uppercase">{activeElement.type}</div>
                    </div>
                  </div>

                  <div className="p-2.5 rounded bg-slate-950 border border-slate-800 text-xs font-mono space-y-1">
                    <div className="text-[10px] text-slate-500">NORMALIZED BOUNDING BOX [0, 1]</div>
                    <div className="text-slate-300">
                      x_min: {activeElement.bbox.x_min.toFixed(4)} • y_min: {activeElement.bbox.y_min.toFixed(4)}
                    </div>
                    <div className="text-slate-300">
                      x_max: {activeElement.bbox.x_max.toFixed(4)} • y_max: {activeElement.bbox.y_max.toFixed(4)}
                    </div>
                  </div>

                  <div className="space-y-1">
                    <span className="text-[10px] font-mono text-slate-500 uppercase">Content Preview</span>
                    <pre className="p-2.5 rounded bg-slate-950 border border-slate-800 text-[11px] font-mono text-slate-300 overflow-x-auto max-h-48 whitespace-pre-wrap">
                      {activeElement.content}
                    </pre>
                  </div>
                </div>
              ) : (
                <div className="text-center p-6 text-xs text-slate-500">
                  Click any element box on the visual document canvas to inspect its AST metadata and coordinates.
                </div>
              )}
            </div>
          )}

          {/* TAB 3: Architectural Guide */}
          {rightTab === 'edu' && (
            <div className="p-4 rounded-xl bg-slate-900/70 border border-slate-800 space-y-4 text-xs text-slate-300 flex-1">
              <h4 className="font-semibold text-slate-100 text-sm flex items-center gap-2">
                <BookOpen className="w-4 h-4 text-cyan-400" />
                Multi-Modal Document Understanding
              </h4>

              <div className="space-y-2">
                <h5 className="font-semibold text-slate-200">1. Why Plain OCR Fails on Enterprise Documents</h5>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  Standard OCR engines discard spatial relationships, turning table rows into arbitrary strings and breaking mathematical fractions.
                  LibraOCR preserves tables as structured matrices and formulas as LaTeX.
                </p>
              </div>

              <div className="space-y-2">
                <h5 className="font-semibold text-slate-200">2. Layout-Aware Semantic Chunking</h5>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  Traditional character-based chunkers slice through table rows mid-sentence.
                  Libra&apos;s LayoutAwareChunker respects tabular boundaries and injects header breadcrumbs so RAG retrieval is 100% grounded.
                </p>
              </div>

              <div className="space-y-2">
                <h5 className="font-semibold text-slate-200">3. Eliminating Hallucinations with Visual Citations</h5>
                <p className="text-slate-400 leading-relaxed text-[11px]">
                  Every generated answer is accompanied by exact spatial coordinates [x_min, y_min, x_max, y_max] and table row references,
                  allowing human operators to instantly audit claims against original document pages.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
