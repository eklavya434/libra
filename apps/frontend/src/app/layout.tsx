import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'https://libraai.me'),
  title: 'Libra | Personal LLM Laboratory & AI Assistant',
  description: 'An educational laboratory and ChatGPT-like assistant built from first principles.',
  alternates: {
    canonical: '/',
  },
  openGraph: {
    title: 'Libra | Personal LLM Laboratory & AI Assistant',
    description: 'An educational laboratory and ChatGPT-like assistant built from first principles.',
    url: 'https://libraai.me',
    siteName: 'Libra',
    type: 'website',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased bg-libra-950 text-slate-100 flex flex-col h-screen overflow-hidden">
        {children}
      </body>
    </html>
  );
}