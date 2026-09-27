import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'https://www.libraai.me'),
  title: 'Libra | Personal LLM Laboratory & AI Assistant',
  description: 'An educational laboratory and ChatGPT-like assistant built from first principles.',
  alternates: {
    canonical: 'https://www.libraai.me',
  },
  openGraph: {
    title: 'Libra | Personal LLM Laboratory & AI Assistant',
    description: 'An educational laboratory and ChatGPT-like assistant built from first principles.',
    url: 'https://www.libraai.me',
    siteName: 'Libra',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Libra | Personal LLM Laboratory & AI Assistant',
    description: 'An educational laboratory and ChatGPT-like assistant built from first principles.',
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