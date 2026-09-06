'use client';

import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Brain, ExternalLink } from 'lucide-react';
import CodeBlock from './CodeBlock';

interface StreamingMarkdownProps {
  content: string;
  isStreaming?: boolean;
}

// Inline formatting parser for bold, italic, inline code, and links
function renderInlineText(text: string): React.ReactNode[] {
  // Matches: `code`, **bold**, *italic*, [text](url)
  const regex = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g;
  const parts: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let key = 0;

  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(<span key={`text-${key++}`}>{text.slice(lastIndex, match.index)}</span>);
    }

    const token = match[0];
    if (token.startsWith('`') && token.endsWith('`')) {
      parts.push(
        <code
          key={`code-${key++}`}
          className="bg-slate-800/80 text-indigo-300 border border-slate-700/60 rounded px-1.5 py-0.5 text-xs font-mono select-all"
        >
          {token.slice(1, -1)}
        </code>
      );
    } else if (token.startsWith('**') && token.endsWith('**')) {
      parts.push(
        <strong key={`bold-${key++}`} className="font-semibold text-slate-100">
          {token.slice(2, -2)}
        </strong>
      );
    } else if (token.startsWith('*') && token.endsWith('*')) {
      parts.push(
        <em key={`italic-${key++}`} className="italic text-slate-200">
          {token.slice(1, -1)}
        </em>
      );
    } else if (token.startsWith('[') && token.includes('](')) {
      const splitIdx = token.indexOf('](');
      const label = token.slice(1, splitIdx);
      const url = token.slice(splitIdx + 2, -1);
      parts.push(
        <a
          key={`link-${key++}`}
          href={url}
          target="_blank"
          rel="noopener noreferrer"
          className="text-indigo-400 hover:text-indigo-300 underline inline-flex items-center gap-0.5"
        >
          {label}
          <ExternalLink className="w-2.5 h-2.5 inline" />
        </a>
      );
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    parts.push(<span key={`text-${key++}`}>{text.slice(lastIndex)}</span>);
  }

  return parts;
}

export default function StreamingMarkdown({ content, isStreaming = false }: StreamingMarkdownProps) {
  const [showThinking, setShowThinking] = useState(true);

  if (!content && isStreaming) {
    return (
      <div className="flex items-center gap-2 text-slate-400 italic text-xs py-1">
        <span className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
        <span>Thinking...</span>
      </div>
    );
  }

  let textToParse = content;
  let thinkContent: string | null = null;
  let isThinkingActive = false;

  // 1. Extract <think>...</think> reasoning blocks
  const thinkEndIdx = textToParse.indexOf('</think>');
  if (textToParse.startsWith('<think>')) {
    if (thinkEndIdx !== -1) {
      thinkContent = textToParse.slice(7, thinkEndIdx).trim();
      textToParse = textToParse.slice(thinkEndIdx + 8).trim();
    } else {
      // In-flight thinking block
      thinkContent = textToParse.slice(7).trim();
      textToParse = '';
      isThinkingActive = true;
    }
  }

  // 2. Parse text into structured sections: Code Fences vs Markdown blocks
  type Segment =
    | { type: 'code'; language: string; code: string; isComplete: boolean }
    | { type: 'markdown'; content: string };

  const segments: Segment[] = [];
  const lines = textToParse.split('\n');
  let inCode = false;
  let currentLang = '';
  let currentCodeLines: string[] = [];
  let currentMdLines: string[] = [];

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.trim().startsWith('```')) {
      if (!inCode) {
        // Opening fence
        if (currentMdLines.length > 0) {
          segments.push({ type: 'markdown', content: currentMdLines.join('\n') });
          currentMdLines = [];
        }
        inCode = true;
        currentLang = line.trim().slice(3).trim() || 'text';
        currentCodeLines = [];
      } else {
        // Closing fence
        segments.push({
          type: 'code',
          language: currentLang,
          code: currentCodeLines.join('\n'),
          isComplete: true,
        });
        inCode = false;
        currentLang = '';
        currentCodeLines = [];
      }
    } else {
      if (inCode) {
        currentCodeLines.push(line);
      } else {
        currentMdLines.push(line);
      }
    }
  }

  // Handle open in-flight code fence during streaming
  if (inCode) {
    segments.push({
      type: 'code',
      language: currentLang,
      code: currentCodeLines.join('\n'),
      isComplete: false,
    });
  } else if (currentMdLines.length > 0) {
    segments.push({ type: 'markdown', content: currentMdLines.join('\n') });
  }

  return (
    <div className="space-y-3 leading-relaxed text-slate-200">
      {/* Collapsible Reasoning / Thinking Block */}
      {thinkContent && (
        <div className="mb-4 rounded-xl border border-slate-800 bg-slate-950/60 overflow-hidden text-xs">
          <button
            type="button"
            onClick={() => setShowThinking(!showThinking)}
            className="w-full flex items-center justify-between px-3.5 py-2 bg-slate-900/60 hover:bg-slate-900 border-b border-slate-850 text-slate-400 font-medium transition-colors"
          >
            <div className="flex items-center gap-2">
              <Brain className={`w-3.5 h-3.5 ${isThinkingActive ? 'text-amber-400 animate-pulse' : 'text-indigo-400'}`} />
              <span className="text-slate-300">
                {isThinkingActive ? 'Thinking in progress...' : 'Reasoning Process'}
              </span>
            </div>
            <div className="flex items-center gap-1.5 text-slate-500">
              <span className="text-[10px]">
                {showThinking ? 'Collapse' : 'Expand'}
              </span>
              {showThinking ? (
                <ChevronDown className="w-3.5 h-3.5" />
              ) : (
                <ChevronRight className="w-3.5 h-3.5" />
              )}
            </div>
          </button>

          {showThinking && (
            <div className="p-3.5 text-slate-400 font-mono text-[11px] leading-relaxed whitespace-pre-wrap border-l-2 border-indigo-500/40 ml-3 my-2 pl-3">
              {thinkContent}
              {isThinkingActive && <span className="inline-block w-1.5 h-3 ml-1 bg-amber-400 animate-pulse align-middle" />}
            </div>
          )}
        </div>
      )}

      {/* Render Markdown Segments */}
      {segments.map((segment, segIdx) => {
        if (segment.type === 'code') {
          return (
            <CodeBlock
              key={`code-${segIdx}`}
              language={segment.language}
              code={segment.code}
              isStreaming={isStreaming && !segment.isComplete}
            />
          );
        }

        // Render Markdown Blocks
        const blockLines = segment.content.split('\n');
        const renderedElements: React.ReactNode[] = [];
        let i = 0;

        while (i < blockLines.length) {
          const line = blockLines[i];
          const trimmed = line.trim();

          if (!trimmed) {
            renderedElements.push(<div key={`empty-${i}`} className="h-2" />);
            i++;
            continue;
          }

          // Headers
          if (trimmed.startsWith('#### ')) {
            renderedElements.push(
              <h4 key={`h4-${i}`} className="text-xs font-semibold text-indigo-300 mt-3 mb-1">
                {renderInlineText(trimmed.slice(5))}
              </h4>
            );
            i++;
          } else if (trimmed.startsWith('### ')) {
            renderedElements.push(
              <h3 key={`h3-${i}`} className="text-sm font-semibold text-slate-100 mt-3.5 mb-1.5">
                {renderInlineText(trimmed.slice(4))}
              </h3>
            );
            i++;
          } else if (trimmed.startsWith('## ')) {
            renderedElements.push(
              <h2 key={`h2-${i}`} className="text-base font-bold text-slate-100 mt-4 mb-2 border-b border-slate-800/80 pb-1">
                {renderInlineText(trimmed.slice(3))}
              </h2>
            );
            i++;
          } else if (trimmed.startsWith('# ')) {
            renderedElements.push(
              <h1 key={`h1-${i}`} className="text-lg font-bold text-slate-100 mt-5 mb-2.5 border-b border-slate-800 pb-1.5">
                {renderInlineText(trimmed.slice(2))}
              </h1>
            );
            i++;
          }
          // Blockquotes
          else if (trimmed.startsWith('> ')) {
            renderedElements.push(
              <blockquote
                key={`bq-${i}`}
                className="border-l-2 border-indigo-500 pl-3.5 my-2 text-slate-400 italic text-xs leading-relaxed"
              >
                {renderInlineText(trimmed.slice(2))}
              </blockquote>
            );
            i++;
          }
          // Lists (unordered)
          else if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
            const listItems: string[] = [];
            while (i < blockLines.length && (blockLines[i].trim().startsWith('- ') || blockLines[i].trim().startsWith('* '))) {
              listItems.push(blockLines[i].trim().slice(2));
              i++;
            }
            renderedElements.push(
              <ul key={`ul-${i}`} className="list-disc list-inside space-y-1 my-2 text-slate-300 pl-1 text-xs">
                {listItems.map((item, lIdx) => (
                  <li key={lIdx} className="leading-relaxed">
                    {renderInlineText(item)}
                  </li>
                ))}
              </ul>
            );
          }
          // Lists (ordered)
          else if (/^\d+\.\s/.test(trimmed)) {
            const listItems: string[] = [];
            while (i < blockLines.length && /^\d+\.\s/.test(blockLines[i].trim())) {
              const match = blockLines[i].trim().match(/^\d+\.\s(.*)/);
              listItems.push(match ? match[1] : blockLines[i].trim());
              i++;
            }
            renderedElements.push(
              <ol key={`ol-${i}`} className="list-decimal list-inside space-y-1 my-2 text-slate-300 pl-1 text-xs">
                {listItems.map((item, lIdx) => (
                  <li key={lIdx} className="leading-relaxed">
                    {renderInlineText(item)}
                  </li>
                ))}
              </ol>
            );
          }
          // Markdown Table
          else if (trimmed.startsWith('|') && trimmed.endsWith('|')) {
            const tableRows: string[][] = [];
            while (i < blockLines.length && blockLines[i].trim().startsWith('|') && blockLines[i].trim().endsWith('|')) {
              const rowLine = blockLines[i].trim();
              // Ignore delimiter line e.g. |---|---|
              if (!rowLine.match(/^\|(?:\s*:?-+:?\s*\|)+$/)) {
                const cells = rowLine
                  .slice(1, -1)
                  .split('|')
                  .map((c) => c.trim());
                tableRows.push(cells);
              }
              i++;
            }

            if (tableRows.length > 0) {
              const headerRow = tableRows[0];
              const bodyRows = tableRows.slice(1);
              renderedElements.push(
                <div key={`table-${i}`} className="overflow-x-auto my-3 rounded-xl border border-slate-800 shadow-sm">
                  <table className="w-full text-left text-xs border-collapse">
                    <thead className="bg-slate-900/90 text-slate-300 border-b border-slate-800">
                      <tr>
                        {headerRow.map((cell, cIdx) => (
                          <th key={cIdx} className="px-3 py-2 font-semibold">
                            {renderInlineText(cell)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-850">
                      {bodyRows.map((row, rIdx) => (
                        <tr key={rIdx} className="hover:bg-slate-900/40 transition-colors">
                          {row.map((cell, cIdx) => (
                            <td key={cIdx} className="px-3 py-2 text-slate-300">
                              {renderInlineText(cell)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              );
            }
          }
          // Standard Paragraph
          else {
            renderedElements.push(
              <p key={`p-${i}`} className="leading-relaxed text-xs">
                {renderInlineText(trimmed)}
              </p>
            );
            i++;
          }
        }

        return <div key={`md-${segIdx}`} className="space-y-1.5">{renderedElements}</div>;
      })}

      {/* Streaming blinking cursor */}
      {isStreaming && (
        <span className="inline-block w-1.5 h-3.5 bg-indigo-400 animate-pulse ml-0.5 align-middle" />
      )}
    </div>
  );
}
