'use client';

import React, { useState, useEffect } from 'react';
import { Database, Plus, Search, Trash2, FileText, Sparkles, CheckCircle2, AlertCircle } from 'lucide-react';
import {
  RAGDocument,
  RAGSearchResult,
  fetchRAGDocuments,
  uploadRAGDocument,
  deleteRAGDocument,
  queryRAG,
} from '@/lib/api';

export default function KnowledgeBaseView() {
  const [documents, setDocuments] = useState<RAGDocument[]>([]);
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<RAGSearchResult[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [notification, setNotification] = useState<string | null>(null);

  const loadDocs = async () => {
    const list = await fetchRAGDocuments();
    setDocuments(list);
  };

  useEffect(() => {
    loadDocs();
  }, []);

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!title.trim() || !content.trim() || loading) return;

    setLoading(true);
    const newDoc = await uploadRAGDocument(title.trim(), content.trim());
    if (newDoc) {
      setTitle('');
      setContent('');
      setNotification(`Successfully indexed "${newDoc.title}" into ${newDoc.chunk_count} vector chunks.`);
      setTimeout(() => setNotification(null), 4000);
      await loadDocs();
    }
    setLoading(false);
  };

  const handleDelete = async (docId: string) => {
    const success = await deleteRAGDocument(docId);
    if (success) {
      setDocuments((prev) => prev.filter((d) => d.id !== docId));
      setSearchResults([]);
    }
  };

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!searchQuery.trim() || searchLoading) return;

    setSearchLoading(true);
    const res = await queryRAG(searchQuery.trim(), 4);
    if (res) {
      setSearchResults(res.results);
    }
    setSearchLoading(false);
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-slate-950 overflow-y-auto p-4 sm:p-8">
      <div className="max-w-6xl mx-auto w-full space-y-6">
        {/* Header Title */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-5">
          <div>
            <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
              <Database className="w-5 h-5 text-indigo-400" />
              <span>Knowledge Base & Semantic RAG</span>
            </h2>
            <p className="text-xs text-slate-400 mt-1">
              Index textbooks and technical notes into vector chunks for real-time grounded inference.
            </p>
          </div>

          <div className="flex items-center gap-2 bg-indigo-950/40 border border-indigo-800/40 px-3 py-1.5 rounded-lg text-xs text-indigo-300">
            <Sparkles className="w-3.5 h-3.5 text-indigo-400" />
            <span>Dense Embeddings (D=128)</span>
          </div>
        </div>

        {notification && (
          <div className="flex items-center gap-2 bg-emerald-950/40 border border-emerald-800/60 text-emerald-300 px-4 py-3 rounded-xl text-xs animate-in fade-in">
            <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>{notification}</span>
          </div>
        )}

        {/* Ingest & Search 2-Column Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Form: Add New Document */}
          <form onSubmit={handleUpload} className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Plus className="w-4 h-4 text-indigo-400" />
              <span>Index New Document</span>
            </h3>

            <div>
              <label className="block text-[11px] font-medium text-slate-400 uppercase tracking-wider mb-1">
                Document Title
              </label>
              <input
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="e.g. Attention Architecture Guide"
                className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-xl px-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all"
              />
            </div>

            <div>
              <label className="block text-[11px] font-medium text-slate-400 uppercase tracking-wider mb-1">
                Content Body
              </label>
              <textarea
                rows={5}
                value={content}
                onChange={(e) => setContent(e.target.value)}
                placeholder="Paste reference text, technical documentation, or factual notes..."
                className="w-full bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-xl p-3 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 transition-all resize-none shadow-inner"
              />
            </div>

            <button
              type="submit"
              disabled={loading || !title.trim() || !content.trim()}
              className="w-full flex items-center justify-center gap-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-40 text-white font-medium py-2 rounded-xl text-xs transition-all shadow-md shadow-indigo-600/20"
            >
              {loading ? 'Chunking & Embedding...' : 'Index & Compute Vectors'}
            </button>
          </form>

          {/* Semantic Retrieval Tester */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm flex flex-col justify-between">
            <div className="space-y-4">
              <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
                <Search className="w-4 h-4 text-cyan-400" />
                <span>Semantic Cosine Search Tester</span>
              </h3>

              <form onSubmit={handleSearch} className="flex gap-2">
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Enter a natural language query..."
                  className="flex-1 bg-slate-950 border border-slate-800 focus:border-indigo-500 rounded-xl px-3 py-2 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                />
                <button
                  type="submit"
                  disabled={searchLoading || !searchQuery.trim()}
                  className="bg-slate-800 hover:bg-slate-750 text-slate-200 font-medium px-4 py-2 rounded-xl text-xs transition-all disabled:opacity-40"
                >
                  {searchLoading ? 'Searching...' : 'Search'}
                </button>
              </form>

              {/* Matched Chunks List */}
              <div className="space-y-2 max-h-[220px] overflow-y-auto pr-1">
                {searchResults.length === 0 ? (
                  <div className="p-6 text-center text-slate-500 text-xs italic">
                    Run a query to inspect matched chunks and cosine similarity rankings.
                  </div>
                ) : (
                  searchResults.map((res) => (
                    <div
                      key={res.chunk.id}
                      className="bg-slate-950/80 border border-slate-800/80 rounded-xl p-3 space-y-1.5"
                    >
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="font-semibold text-slate-300 truncate">
                          #{res.rank} {res.chunk.doc_title}
                        </span>
                        <span className="text-indigo-400 font-mono font-medium">
                          {(res.score * 100).toFixed(1)}% match
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                        {res.chunk.text}
                      </p>
                    </div>
                  ))
                )}
              </div>
            </div>

            <div className="text-[10px] text-slate-500 border-t border-slate-850 pt-2">
              Formula: cos(u, v) = u • v / (||u|| ||v||) with L2 normalized dense vectors.
            </div>
          </div>
        </div>

        {/* Indexed Documents Table */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 space-y-4 shadow-sm">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <FileText className="w-4 h-4 text-emerald-400" />
              <span>Indexed Knowledge Base Documents</span>
            </h3>
            <span className="text-xs text-slate-400 font-mono">
              Total: {documents.length}
            </span>
          </div>

          {documents.length === 0 ? (
            <div className="py-8 text-center text-slate-500 text-xs italic">
              No documents indexed in vector memory yet. Add your first document above.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {documents.map((doc) => (
                <div
                  key={doc.id}
                  className="bg-slate-950/70 border border-slate-800 rounded-xl p-3.5 flex flex-col justify-between hover:border-slate-700 transition-all group"
                >
                  <div className="space-y-1">
                    <div className="flex items-center justify-between">
                      <h4 className="text-xs font-semibold text-slate-200 truncate">{doc.title}</h4>
                      <button
                        onClick={() => handleDelete(doc.id)}
                        className="opacity-0 group-hover:opacity-100 hover:text-rose-400 text-slate-500 transition-opacity p-0.5"
                        title="Delete document"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
                      {doc.content}
                    </p>
                  </div>

                  <div className="mt-3 pt-2 border-t border-slate-850 flex items-center justify-between text-[10px] text-slate-500">
                    <span>{doc.chunk_count} vector chunks</span>
                    <span>{new Date(doc.created_at).toLocaleDateString()}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
