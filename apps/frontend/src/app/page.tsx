'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from '@/components/Sidebar';
import ChatArea from '@/components/ChatArea';
import ArenaView from '@/components/ArenaView';
import KnowledgeBaseView from '@/components/KnowledgeBaseView';
import {
  ConversationSummary,
  fetchConversations,
  createConversation,
  deleteConversation,
} from '@/lib/api';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'chat' | 'arena' | 'rag'>('chat');
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string>('');

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
    const newConv = await createConversation('New Conversation', 'libra-llama-tied');
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
      <Sidebar
        currentSessionId={currentSessionId}
        activeTab={activeTab}
        conversations={conversations}
        onSelectTab={setActiveTab}
        onSelectConversation={(id) => {
          setCurrentSessionId(id);
          setActiveTab('chat');
        }}
        onNewConversation={handleNewConversation}
        onDeleteConversation={handleDeleteConversation}
      />
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        {activeTab === 'chat' && (
          <ChatArea
            conversationId={currentSessionId}
            onConversationUpdated={loadConversations}
          />
        )}
        {activeTab === 'arena' && <ArenaView />}
        {activeTab === 'rag' && <KnowledgeBaseView />}
      </main>
    </div>
  );
}