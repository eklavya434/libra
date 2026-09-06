'use client';

import Sidebar from '@/components/Sidebar';
import ChatArea from '@/components/ChatArea';

export default function Home() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-950">
      <Sidebar currentSessionId="session-0" />
      <main className="flex-1 flex flex-col h-full overflow-hidden">
        <ChatArea />
      </main>
    </div>
  );
}