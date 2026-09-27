'use client';

import React, { useEffect, useState, useCallback, useRef } from 'react';
import { Cpu, HardDrive, ShieldCheck, AlertCircle, RefreshCw, CheckCircle2 } from 'lucide-react';
import { API_BASE_URL, libraFetch } from '@/lib/api';

interface HardwareData {
  cpu_model: string;
  cpu_physical_cores: number;
  cpu_logical_cores: number;
  ram_total_gb: number;
  ram_available_gb: number;
  disk_free_gb: number;
  device_tier: string;
  has_cuda: boolean;
}

interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  learning_mode: boolean;
  hardware: HardwareData;
}

type BadgeStatus = 'connecting' | 'slow' | 'connected' | 'error';

export default function StatusBadge() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [status, setStatus] = useState<BadgeStatus>('connecting');
  const slowTimerRef = useRef<NodeJS.Timeout | null>(null);

  const fetchHealth = useCallback(async () => {
    setStatus((prev) => (prev === 'connected' ? 'connected' : 'connecting'));

    // If connecting takes > 5s, transition to 'slow' (waking up cold container)
    if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
    slowTimerRef.current = setTimeout(() => {
      setStatus((prev) => (prev === 'connecting' ? 'slow' : prev));
    }, 5000);

    try {
      const res = await libraFetch(`${API_BASE_URL}/api/v1/health`);
      if (!res.ok) throw new Error('API offline');
      const json = await res.json();
      if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
      setData(json);
      setStatus('connected');
    } catch {
      if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
      setStatus('error');
      setData(null);
    }
  }, []);

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => {
      clearInterval(interval);
      if (slowTimerRef.current) clearTimeout(slowTimerRef.current);
    };
  }, [fetchHealth]);

  if (status === 'connecting') {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/60 border border-slate-700/50 text-xs text-slate-400">
        <RefreshCw className="w-3 h-3 animate-spin text-indigo-400" />
        <span>Connecting to Libra Backend...</span>
      </div>
    );
  }

  if (status === 'slow') {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-950/40 border border-amber-800/60 text-xs text-amber-300">
        <RefreshCw className="w-3 h-3 animate-spin text-amber-400" />
        <span>Waking up backend (~30s on free tier)...</span>
      </div>
    );
  }

  if (status === 'error' || !data) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-950/40 border border-amber-800/60 text-xs text-amber-300">
        <AlertCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
        <span>Backend asleep or redeploying</span>
        <button
          onClick={fetchHealth}
          className="ml-1 inline-flex items-center gap-1 rounded bg-amber-500/20 px-2 py-0.5 text-[10px] font-semibold text-amber-200 hover:bg-amber-500/30 transition-colors"
        >
          <RefreshCw className="w-2.5 h-2.5" />
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 text-xs bg-slate-900/80 border border-slate-800 rounded-lg px-3 py-1.5 shadow-sm">
      <div className="flex items-center gap-1.5 text-emerald-400 font-medium">
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
        </span>
        <span>v{data.version} Online</span>
      </div>

      <div className="h-3 w-px bg-slate-700" />

      <div
        className="flex items-center gap-1 text-slate-300"
        title={`Physical Cores: ${data.hardware?.cpu_physical_cores ?? 4}, Logical: ${data.hardware?.cpu_logical_cores ?? 8}`}
      >
        <Cpu className="w-3.5 h-3.5 text-indigo-400" />
        <span>{data.hardware?.cpu_physical_cores ?? 4}C CPU</span>
      </div>

      <div
        className="flex items-center gap-1 text-slate-300"
        title={`Free Disk: ${data.hardware?.disk_free_gb ?? 70} GB`}
      >
        <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
        <span>
          {data.hardware?.ram_available_gb ?? 8}/{data.hardware?.ram_total_gb ?? 16} GB RAM
        </span>
      </div>

      <div className="flex items-center gap-1 text-slate-400">
        <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
        <span className="capitalize">{data.hardware?.device_tier ?? 'cpu-light'}</span>
      </div>
    </div>
  );
}