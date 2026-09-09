'use client';

import React, { useState, useEffect, useCallback } from 'react';
import {
  Play,
  RotateCcw,
  Plus,
  Trash2,
  ChevronUp,
  ChevronDown,
  Terminal,
  Database,
  BarChart2,
  Code2,
  FileText,
  Clock,
  Sparkles,
  Layers,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';

interface VariableItem {
  name: string;
  type: string;
  value_repr: string;
  size?: string | null;
}

interface CellOutput {
  status: 'ok' | 'error';
  execution_count: number;
  stdout: string;
  stderr: string;
  result: string | null;
  mime_outputs: Record<string, string>;
  duration_ms: number;
  error_message?: string | null;
}

interface NotebookCell {
  id: string;
  type: 'code' | 'markdown';
  content: string;
  output?: CellOutput | null;
  isRunning?: boolean;
}

interface NotebookPreset {
  id: string;
  title: string;
  description: string;
  cells: {
    type: 'code' | 'markdown';
    content: string;
  }[];
}

export default function NotebookView() {
  const [sessionId, setSessionId] = useState<string>('default-session');
  const [cells, setCells] = useState<NotebookCell[]>([
    {
      id: 'cell-1',
      type: 'markdown',
      content: '### 🚀 Libra Interactive Notebook & Data Analytics Sandbox\nExecute Python code with stateful variables, tabular analysis (`LibraTable`), and vector SVG charts (`LibraChart`).',
    },
    {
      id: 'cell-2',
      type: 'code',
      content: '# Define sample data and calculate sum\nx = 15\ny = 25\nprint(f"Computed sum: {x + y}")\nx + y',
    },
  ]);
  const [variables, setVariables] = useState<VariableItem[]>([]);
  const [presets, setPresets] = useState<NotebookPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = useState<string>('');
  const [isKernelRunning, setIsKernelRunning] = useState<boolean>(false);
  const [showVarDrawer, setShowVarDrawer] = useState<boolean>(true);
  const [activeCellId, setActiveCellId] = useState<string>('cell-2');

  // Fetch presets and variables on mount
  const fetchPresets = useCallback(async () => {
    try {
      const res = await fetch('/api/v1/notebook/presets');
      if (res.ok) {
        const data = await res.json();
        setPresets(data);
      }
    } catch {
      // Backend not running or offline
    }
  }, []);

  const fetchVariables = useCallback(async (sid: string) => {
    try {
      const res = await fetch(`/api/v1/notebook/sessions/${sid}/variables`);
      if (res.ok) {
        const data = await res.json();
        setVariables(data);
      }
    } catch {
      // Offline fallback
    }
  }, []);

  useEffect(() => {
    fetchPresets();
    fetchVariables(sessionId);
  }, [fetchPresets, fetchVariables, sessionId]);

  // Load a preset notebook
  const loadPreset = (presetId: string) => {
    const p = presets.find((item) => item.id === presetId);
    if (!p) return;
    const newCells: NotebookCell[] = p.cells.map((c, idx) => ({
      id: `cell-${Date.now()}-${idx}`,
      type: c.type,
      content: c.content,
      output: null,
      isRunning: false,
    }));
    setCells(newCells);
    setSelectedPreset(presetId);
  };

  // Run a single cell
  const runCell = async (cellId: string) => {
    const targetIndex = cells.findIndex((c) => c.id === cellId);
    if (targetIndex === -1) return;
    const cell = cells[targetIndex];
    if (cell.type === 'markdown') return;

    setCells((prev) =>
      prev.map((c) => (c.id === cellId ? { ...c, isRunning: true } : c))
    );

    try {
      const res = await fetch(`/api/v1/notebook/sessions/${sessionId}/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code: cell.content }),
      });

      if (res.ok) {
        const output: CellOutput = await res.json();
        setCells((prev) =>
          prev.map((c) => (c.id === cellId ? { ...c, output, isRunning: false } : c))
        );
        fetchVariables(sessionId);
      } else {
        const errData = await res.json().catch(() => ({ detail: 'Execution failed' }));
        setCells((prev) =>
          prev.map((c) =>
            c.id === cellId
              ? {
                  ...c,
                  isRunning: false,
                  output: {
                    status: 'error',
                    execution_count: 0,
                    stdout: '',
                    stderr: errData.detail || 'Execution error',
                    result: null,
                    mime_outputs: {},
                    duration_ms: 0,
                    error_message: errData.detail,
                  },
                }
              : c
          )
        );
      }
    } catch (e: any) {
      setCells((prev) =>
        prev.map((c) =>
          c.id === cellId
            ? {
                ...c,
                isRunning: false,
                output: {
                  status: 'error',
                  execution_count: 0,
                  stdout: '',
                  stderr: e.message || 'Network error',
                  result: null,
                  mime_outputs: {},
                  duration_ms: 0,
                  error_message: e.message,
                },
              }
            : c
        )
      );
    }
  };

  // Run all cells sequentially
  const runAllCells = async () => {
    setIsKernelRunning(true);
    for (const cell of cells) {
      if (cell.type === 'code') {
        await runCell(cell.id);
      }
    }
    setIsKernelRunning(false);
  };

  // Restart kernel
  const restartKernel = async () => {
    try {
      await fetch(`/api/v1/notebook/sessions/${sessionId}/reset`, { method: 'POST' });
      setCells((prev) => prev.map((c) => ({ ...c, output: null, isRunning: false })));
      setVariables([]);
    } catch {
      // Offline fallback
    }
  };

  // Add Cell
  const addCell = (type: 'code' | 'markdown', afterId?: string) => {
    const newCell: NotebookCell = {
      id: `cell-${Date.now()}`,
      type,
      content: type === 'code' ? '# Write Python code here...\n' : '### New Section\nDescribe your analysis here.',
      output: null,
    };

    if (!afterId) {
      setCells((prev) => [...prev, newCell]);
    } else {
      const idx = cells.findIndex((c) => c.id === afterId);
      const updated = [...cells];
      updated.splice(idx + 1, 0, newCell);
      setCells(updated);
    }
    setActiveCellId(newCell.id);
  };

  // Delete Cell
  const deleteCell = (cellId: string) => {
    if (cells.length <= 1) return;
    setCells((prev) => prev.filter((c) => c.id !== cellId));
  };

  // Move Cell
  const moveCell = (cellId: string, direction: 'up' | 'down') => {
    const idx = cells.findIndex((c) => c.id === cellId);
    if (idx === -1) return;
    if (direction === 'up' && idx === 0) return;
    if (direction === 'down' && idx === cells.length - 1) return;

    const newCells = [...cells];
    const targetIdx = direction === 'up' ? idx - 1 : idx + 1;
    const temp = newCells[idx];
    newCells[idx] = newCells[targetIdx];
    newCells[targetIdx] = temp;
    setCells(newCells);
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans">
      {/* Main Notebook Canvas */}
      <div className="flex-1 flex flex-col min-w-0 border-r border-slate-800">
        {/* Top Control Bar */}
        <header className="flex flex-wrap items-center justify-between gap-3 px-6 py-3 border-b border-slate-800 bg-slate-900/60 backdrop-blur-sm">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-400">
              <Terminal className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-semibold text-white tracking-wide">
                  LibraNotebook
                </h1>
                <span className="px-2 py-0.5 text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 rounded-full">
                  Kernel: Ready (Python 3.11+)
                </span>
              </div>
              <p className="text-xs text-slate-400">
                Stateful Code Interpreter & First-Principles Data Analytics
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Presets Dropdown */}
            <select
              value={selectedPreset}
              onChange={(e) => loadPreset(e.target.value)}
              className="bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-indigo-500"
            >
              <option value="">Choose an educational preset...</option>
              {presets.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title}
                </option>
              ))}
            </select>

            {/* Run All Button */}
            <button
              onClick={runAllCells}
              disabled={isKernelRunning}
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium rounded-lg bg-indigo-600 hover:bg-indigo-500 text-white transition-colors shadow-sm disabled:opacity-50"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>Run All</span>
            </button>

            {/* Restart Kernel */}
            <button
              onClick={restartKernel}
              className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors border border-slate-700"
              title="Restart session & reset memory"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Restart</span>
            </button>

            {/* Add Cell Buttons */}
            <button
              onClick={() => addCell('code')}
              className="flex items-center gap-1 px-2 py-1.5 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>Code</span>
            </button>

            <button
              onClick={() => addCell('markdown')}
              className="flex items-center gap-1 px-2 py-1.5 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700"
            >
              <FileText className="w-3.5 h-3.5" />
              <span>Markdown</span>
            </button>

            {/* Toggle Variable Explorer */}
            <button
              onClick={() => setShowVarDrawer(!showVarDrawer)}
              className={`p-1.5 rounded-lg border text-xs transition-colors ${
                showVarDrawer
                  ? 'bg-indigo-600/20 border-indigo-500/40 text-indigo-300'
                  : 'bg-slate-800 border-slate-700 text-slate-400 hover:text-slate-200'
              }`}
              title="Toggle Variable Explorer"
            >
              <Database className="w-4 h-4" />
            </button>
          </div>
        </header>

        {/* Cells Scroll Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {cells.map((cell, index) => (
            <div
              key={cell.id}
              onClick={() => setActiveCellId(cell.id)}
              className={`group relative rounded-xl border transition-all ${
                activeCellId === cell.id
                  ? 'border-indigo-500/50 bg-slate-900/70 shadow-lg shadow-indigo-950/20'
                  : 'border-slate-800/80 bg-slate-900/30 hover:border-slate-700'
              }`}
            >
              {/* Cell Header / Action bar */}
              <div className="flex items-center justify-between px-3 py-1.5 border-b border-slate-800/60 bg-slate-950/40 text-[11px] text-slate-400">
                <div className="flex items-center gap-2">
                  <span className="font-mono font-semibold text-slate-500">
                    {cell.type === 'code' ? (
                      cell.output?.execution_count ? (
                        <span className="text-indigo-400">[{cell.output.execution_count}]</span>
                      ) : cell.isRunning ? (
                        <span className="text-amber-400">[*]</span>
                      ) : (
                        <span>[ ]</span>
                      )
                    ) : (
                      <span className="text-slate-500 font-sans uppercase text-[9px] tracking-wider">MD</span>
                    )}
                  </span>
                  <span className="text-slate-500">Cell #{index + 1}</span>
                  {cell.output?.duration_ms ? (
                    <span className="flex items-center gap-1 text-[10px] text-slate-500">
                      <Clock className="w-2.5 h-2.5" />
                      {cell.output.duration_ms} ms
                    </span>
                  ) : null}
                </div>

                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  {cell.type === 'code' && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        runCell(cell.id);
                      }}
                      disabled={cell.isRunning}
                      className="p-1 rounded hover:bg-slate-800 text-slate-300 hover:text-emerald-400 transition-colors"
                      title="Run Cell (Shift+Enter)"
                    >
                      <Play className="w-3 h-3 fill-current" />
                    </button>
                  )}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      moveCell(cell.id, 'up');
                    }}
                    className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200"
                    title="Move Up"
                  >
                    <ChevronUp className="w-3 h-3" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      moveCell(cell.id, 'down');
                    }}
                    className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200"
                    title="Move Down"
                  >
                    <ChevronDown className="w-3 h-3" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteCell(cell.id);
                    }}
                    className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-rose-400"
                    title="Delete Cell"
                  >
                    <Trash2 className="w-3 h-3" />
                  </button>
                </div>
              </div>

              {/* Cell Input Area */}
              <div className="p-3">
                {cell.type === 'code' ? (
                  <div className="relative">
                    <textarea
                      value={cell.content}
                      onChange={(e) => {
                        const val = e.target.value;
                        setCells((prev) =>
                          prev.map((c) => (c.id === cell.id ? { ...c, content: val } : c))
                        );
                      }}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' && e.shiftKey) {
                          e.preventDefault();
                          runCell(cell.id);
                        }
                      }}
                      rows={Math.max(2, cell.content.split('\n').length)}
                      className="w-full bg-slate-950 text-slate-200 font-mono text-xs rounded-lg p-3 border border-slate-800 focus:outline-none focus:border-indigo-500/80 resize-y leading-relaxed tracking-tight"
                      placeholder="# Write Python code here..."
                    />
                    <div className="absolute right-3 bottom-3 text-[10px] text-slate-600 font-mono pointer-events-none">
                      Shift + Enter to run
                    </div>
                  </div>
                ) : (
                  <textarea
                    value={cell.content}
                    onChange={(e) => {
                      const val = e.target.value;
                      setCells((prev) =>
                        prev.map((c) => (c.id === cell.id ? { ...c, content: val } : c))
                      );
                    }}
                    rows={Math.max(2, cell.content.split('\n').length)}
                    className="w-full bg-slate-950/60 text-slate-300 text-xs rounded-lg p-3 border border-slate-800 focus:outline-none focus:border-indigo-500/80 resize-y leading-relaxed"
                    placeholder="Enter Markdown notes or documentation..."
                  />
                )}
              </div>

              {/* Cell Output Area */}
              {cell.output && (
                <div className="border-t border-slate-800/80 bg-slate-950/60 p-4 space-y-3 rounded-b-xl">
                  {/* Stdout */}
                  {cell.output.stdout && (
                    <div className="font-mono text-xs text-slate-300 bg-slate-900/60 p-3 rounded-lg border border-slate-800 whitespace-pre-wrap leading-relaxed">
                      {cell.output.stdout}
                    </div>
                  )}

                  {/* Stderr or Error */}
                  {cell.output.stderr && (
                    <div className="flex items-start gap-2 text-xs font-mono text-rose-400 bg-rose-950/20 border border-rose-900/40 p-3 rounded-lg whitespace-pre-wrap">
                      <AlertTriangle className="w-4 h-4 shrink-0 text-rose-500 mt-0.5" />
                      <div>{cell.output.stderr}</div>
                    </div>
                  )}

                  {/* Rich SVG Charts */}
                  {cell.output.mime_outputs['image/svg+xml'] && (
                    <div className="flex justify-center p-2 bg-slate-900/40 rounded-lg border border-slate-800 overflow-x-auto">
                      <div
                        dangerouslySetInnerHTML={{
                          __html: cell.output.mime_outputs['image/svg+xml'],
                        }}
                      />
                    </div>
                  )}

                  {/* Rich HTML Table */}
                  {cell.output.mime_outputs['text/html'] && (
                    <div
                      dangerouslySetInnerHTML={{
                        __html: cell.output.mime_outputs['text/html'],
                      }}
                    />
                  )}

                  {/* Expression Return Value (Out [x]) */}
                  {cell.output.result && !cell.output.mime_outputs['text/html'] && !cell.output.mime_outputs['image/svg+xml'] && (
                    <div className="flex items-baseline gap-2 font-mono text-xs">
                      <span className="text-indigo-400 text-[11px] font-semibold">
                        Out [{cell.output.execution_count}]:
                      </span>
                      <span className="text-slate-200">{cell.output.result}</span>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Right Drawer: Live Variable Explorer */}
      {showVarDrawer && (
        <aside className="w-80 bg-slate-900/40 flex flex-col shrink-0 border-l border-slate-800 overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800 bg-slate-950/50">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-indigo-400" />
              <h2 className="text-xs font-semibold text-slate-200 uppercase tracking-wider">
                Variable Explorer
              </h2>
            </div>
            <button
              onClick={() => fetchVariables(sessionId)}
              className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200"
              title="Refresh variables"
            >
              <RefreshCw className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-2">
            {variables.length === 0 ? (
              <div className="text-center py-12 px-4 text-slate-400 text-xs">
                <Code2 className="w-8 h-8 mx-auto mb-2 opacity-30 text-slate-500" />
                <p className="font-medium text-slate-400">No active variables</p>
                <p className="text-[11px] text-slate-400 mt-1">
                  Run a code cell assigning variables (e.g. <code className="text-indigo-400">x = 42</code>) to inspect session memory.
                </p>
              </div>
            ) : (
              variables.map((v) => (
                <div
                  key={v.name}
                  className="p-2.5 rounded-lg border border-slate-800 bg-slate-950/60 hover:border-slate-700 transition-colors"
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-xs font-semibold text-indigo-300">
                      {v.name}
                    </span>
                    <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                      {v.type}
                    </span>
                  </div>
                  {v.size && (
                    <div className="text-[10px] text-slate-400 font-mono mb-1">
                      {v.size}
                    </div>
                  )}
                  <div className="text-[11px] font-mono text-slate-400 bg-slate-900/70 p-1.5 rounded border border-slate-800/80 truncate">
                    {v.value_repr}
                  </div>
                </div>
              ))
            )}
          </div>

          <div className="p-3 border-t border-slate-800 bg-slate-950/60 text-[11px] text-slate-400 space-y-1">
            <div className="flex justify-between">
              <span>Total Variables:</span>
              <span className="font-mono text-slate-300">{variables.length}</span>
            </div>
            <div className="flex justify-between">
              <span>Session ID:</span>
              <span className="font-mono text-slate-300">{sessionId}</span>
            </div>
          </div>
        </aside>
      )}
    </div>
  );
}
