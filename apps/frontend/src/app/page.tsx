'use client';

import React, { useState } from 'react';
import Sidebar from '@/components/Sidebar';
import ChatArea from '@/components/ChatArea';
import ArenaView from '@/components/ArenaView';

export default function Home() {
  const [activeTab, setActiveTab] = useState<'chat' | 'arena'>('chat');

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950">
      <Sidebar
        currentSessionId="session-0"
        activeTab={activeTab}
        onSelectTab={setActiveTab}
      />
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        {activeTab === 'chat' ? <ChatArea /> : <ArenaView />}
      </main>
    </div>
  );
}