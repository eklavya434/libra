'use client';

import React, { useState } from 'react';
import { Check, Copy, Terminal } from 'lucide-react';

interface CodeBlockProps {
  language?: string;
  code: string;
  isStreaming?: boolean;
}

// Lightweight syntax highlighting token patterns
const KEYWORDS = new Set([
  'def', 'class', 'return', 'import', 'from', 'as', 'if', 'elif', 'else',
  'for', 'while', 'in', 'is', 'not', 'and', 'or', 'try', 'except', 'finally',
  'with', 'lambda', 'yield', 'async', 'await', 'const', 'let', 'var', 'function',
  'export', 'default', 'extends', 'implements', 'interface', 'type', 'new', 'this',
  'select', 'where', 'insert', 'into', 'update', 'delete', 'create', 'table', 'drop',
  'fn', 'pub', 'struct', 'enum', 'impl', 'mut', 'match',
]);

const BUILTINS = new Set([
  'print', 'len', 'range', 'int', 'str', 'float', 'bool', 'list', 'dict', 'set',
  'tuple', 'None', 'True', 'False', 'null', 'undefined', 'console', 'log', 'error',
  'self', 'super', 'Promise', 'Array', 'Object', 'String', 'Number', 'Boolean',
]);

function highlightLine(line: string, lang: string): React.ReactNode[] {
  // Regex to match comments, strings, words/identifiers, numbers, or punctuation
  const tokenRegex = /(\/\/[^\n]*|#[^\n]*|--[^\n]*|"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`|\b[a-zA-Z_]\w*\b|\b\d+(?:\.\d+)?\b|[^\s\w]+|\s+)/g;

  const nodes: React.ReactNode[] = [];
  let match: RegExpExecArray | null;
  let keyIndex = 0;

  while ((match = tokenRegex.exec(line)) !== null) {
    const text = match[0];
    const key = `${keyIndex++}`;

    if (text.startsWith('#') || text.startsWith('//') || text.startsWith('--')) {
      nodes.push(<span key={key} className="text-slate-500 italic">{text}</span>);
    } else if (text.startsWith('"') || text.startsWith("'") || text.startsWith('`')) {
      nodes.push(<span key={key} className="text-emerald-400">{text}</span>);
    } else if (/^\d+(\.\d+)?$/.test(text)) {
      nodes.push(<span key={key} className="text-amber-400">{text}</span>);
    } else if (KEYWORDS.has(text.toLowerCase()) || KEYWORDS.has(text)) {
      nodes.push(<span key={key} className="text-purple-400 font-medium">{text}</span>);
    } else if (BUILTINS.has(text)) {
      nodes.push(<span key={key} className="text-cyan-400">{text}</span>);
    } else {
      nodes.push(<span key={key} className="text-slate-200">{text}</span>);
    }
  }

  return nodes.length > 0 ? nodes : [<span key="empty">{line}</span>];
}

export default function CodeBlock({ language = 'text', code, isStreaming = false }: CodeBlockProps) {
  const [copied, setCopied] = useState(false);

  const cleanLang = (language || 'text').trim().toLowerCase();
  const lines = code.split('\n');

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="my-3.5 rounded-xl border border-slate-800 bg-slate-950 overflow-hidden shadow-lg shadow-black/40 group">
      {/* Code Header Bar */}
      <div className="flex items-center justify-between px-4 py-1.5 bg-slate-900/90 border-b border-slate-850 text-xs text-slate-400 select-none">
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5 text-indigo-400" />
          <span className="font-mono text-[11px] uppercase tracking-wider text-slate-300 font-medium">
            {cleanLang}
          </span>
          {isStreaming && (
            <span className="flex items-center gap-1 text-[10px] text-indigo-400 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-indigo-400 animate-pulse" />
              <span>generating...</span>
            </span>
          )}
        </div>

        <button
          onClick={handleCopy}
          type="button"
          className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-slate-200 bg-slate-800/60 hover:bg-slate-800 px-2 py-0.5 rounded-md transition-colors"
          title="Copy code"
        >
          {copied ? (
            <>
              <Check className="w-3 h-3 text-emerald-400" />
              <span className="text-emerald-400">Copied!</span>
            </>
          ) : (
            <>
              <Copy className="w-3 h-3" />
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Code Body */}
      <div className="overflow-x-auto p-4 text-xs font-mono leading-relaxed selection:bg-indigo-900/60">
        <pre className="text-slate-200">
          <code>
            {lines.map((line, idx) => (
              <div key={idx} className="table-row">
                <span className="table-cell select-none pr-4 text-right text-slate-600 text-[10px] w-6">
                  {idx + 1}
                </span>
                <span className="table-cell whitespace-pre">
                  {highlightLine(line, cleanLang)}
                </span>
              </div>
            ))}
          </code>
        </pre>
      </div>
    </div>
  );
}
