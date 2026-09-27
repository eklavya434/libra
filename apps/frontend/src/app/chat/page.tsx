'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from '@/components/Sidebar';
import ChatArea from '@/components/ChatArea';
import ArenaView from '@/components/ArenaView';
import KnowledgeBaseView from '@/components/KnowledgeBaseView';
import CorpusViewer from '@/components/CorpusViewer';
import SecurityInspector from '@/components/SecurityInspector';
import ObservabilityView from '@/components/ObservabilityView';
import BatchInferenceView from '@/components/BatchInferenceView';
import DocumentOCRView from '@/components/DocumentOCRView';
import NotebookView from '@/components/NotebookView';
import LongContextView from '@/components/LongContextView';
import MCTSTreeView from '@/components/MCTSTreeView';
import DistillationView from '@/components/DistillationView';
import MoEView from '@/components/MoEView';
import MedusaView from '@/components/MedusaView';
import KTOView from '@/components/KTOView';
import VerifiableSearchView from '@/components/VerifiableSearchView';
import SelfRewardingView from '@/components/SelfRewardingView';
import GrandCapstoneView from '@/components/GrandCapstoneView';
import OnboardingTour from '@/components/OnboardingTour';
import {
  ConversationSummary,
  fetchConversations,
  createConversation,
  deleteConversation,
  fetchDefaultModel,
} from '@/lib/api';

export default function ChatApp() {
  const [activeTab, setActiveTab] = useState<'chat' | 'arena' | 'rag' | 'corpus' | 'security' | 'observability' | 'batch' | 'document' | 'notebook' | 'long_context' | 'mcts' | 'distillation' | 'moe' | 'medusa' | 'kto' | 'verifiable_search' | 'self_rewarding' | 'grand_capstone'>('chat');
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');
  const [configuredDefaultModel, setConfiguredDefaultModel] = useState<string | null>(null);
  // On small screens the sidebar is an overlay drawer; docked full-height on md+.
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    fetchDefaultModel().then(setConfiguredDefaultModel);
  }, []);

  const loadConversations = useCallback(async () => {
    const list = await fetchConversations();
    setConversations(list);
    if (list.length > 0 && !currentSessionId) {
      setCurrentSessionId(list[0].id);
    }
  }, [currentSessionId]);

  useEffect(() => {
    loadConversations();
  }, [loadConversations]);

  const handleNewConversation = async () => {
    const newConv = await createConversation(
      'New Conversation',
      configuredDefaultModel ?? undefined
    );
    if (newConv) {
      setConversations((prev) => [newConv, ...prev]);
      setCurrentSessionId(newConv.id);
      setActiveTab('chat');
    }
  };

  const handleDeleteConversation = async (id: string) => {
    const success = await deleteConversation(id);
    if (success) {
      const remaining = conversations.filter((c) => c.id !== id);
      setConversations(remaining);
      if (currentSessionId === id) {
        if (remaining.length > 0) {
          setCurrentSessionId(remaining[0].id);
        } else {
          setCurrentSessionId('');
        }
      }
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950">
      {/* Mobile drawer overlay */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-slate-950/70 backdrop-blur-sm md:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar: docked on md+, animated overlay drawer on small screens */}
      <div
        className={`fixed inset-y-0 left-0 z-40 flex-shrink-0 transform transition-transform duration-200 md:static md:translate-x-0 ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        <Sidebar
          currentSessionId={currentSessionId}
          activeTab={activeTab}
          conversations={conversations}
          onSelectTab={(tab) => {
            setActiveTab(tab);
            setSidebarOpen(false);
          }}
          onSelectConversation={(id) => {
            setCurrentSessionId(id);
            setActiveTab('chat');
            setSidebarOpen(false);
          }}
          onNewConversation={() => {
            handleNewConversation();
            setSidebarOpen(false);
          }}
          onDeleteConversation={handleDeleteConversation}
        />
      </div>
      <main className="flex h-full flex-1 flex-col overflow-hidden">
        {activeTab === 'chat' && (
          <ChatArea
            conversationId={currentSessionId}
            onConversationUpdated={loadConversations}
            onSelectConversation={setCurrentSessionId}
            onToggleSidebar={() => setSidebarOpen(true)}
          />
        )}
        {activeTab === 'arena' && <ArenaView />}
        {activeTab === 'rag' && <KnowledgeBaseView />}
        {activeTab === 'corpus' && <CorpusViewer />}
        {activeTab === 'security' && <SecurityInspector />}
        {activeTab === 'observability' && <ObservabilityView />}
        {activeTab === 'batch' && <BatchInferenceView />}
        {activeTab === 'document' && <DocumentOCRView />}
        {activeTab === 'notebook' && <NotebookView />}
        {activeTab === 'long_context' && <LongContextView />}
        {activeTab === 'mcts' && <MCTSTreeView />}
        {activeTab === 'distillation' && <DistillationView />}
        {activeTab === 'moe' && <MoEView />}
        {activeTab === 'medusa' && <MedusaView />}
        {activeTab === 'kto' && <KTOView />}
        {activeTab === 'verifiable_search' && <VerifiableSearchView />}
        {activeTab === 'self_rewarding' && <SelfRewardingView />}
        {activeTab === 'grand_capstone' && <GrandCapstoneView />}
      </main>
      <OnboardingTour />
    </div>
  );
}