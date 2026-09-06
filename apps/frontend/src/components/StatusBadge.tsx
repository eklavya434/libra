'use client';

import React, { useEffect, useState } from 'react';
import { Cpu, HardDrive, ShieldCheck, AlertCircle, RefreshCw } from 'lucide-react';

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

export default function StatusBadge() {
  const [data, setData] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  const fetchHealth = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/api/v1/health');
      if (!res.ok) throw new Error('API offline');
      const json = await res.json();
      setData(json);
      setError(false);
    } catch {
      setError(true);
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/60 border border-slate-700/50 text-xs text-slate-400">
        <RefreshCw className="w-3 h-3 animate-spin text-indigo-400" />
        <span>Connecting to Libra Backend...</span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-amber-950/40 border border-amber-800/60 text-xs text-amber-300">
        <AlertCircle className="w-3.5 h-3.5 text-amber-400" />
        <span>Backend Standby (:8000)</span>
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

      <div className="flex items-center gap-1 text-slate-300" title={`Physical Cores: ${data.hardware.cpu_physical_cores}, Logical: ${data.hardware.cpu_logical_cores}`}>
        <Cpu className="w-3.5 h-3.5 text-indigo-400" />
        <span>{data.hardware.cpu_physical_cores}C CPU</span>
      </div>

      <div className="flex items-center gap-1 text-slate-300" title={`Free Disk: ${data.hardware.disk_free_gb} GB`}>
        <HardDrive className="w-3.5 h-3.5 text-cyan-400" />
        <span>{data.hardware.ram_available_gb}/{data.hardware.ram_total_gb} GB RAM</span>
      </div>

      <div className="flex items-center gap-1 text-slate-400">
        <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
        <span className="capitalize">{data.hardware.device_tier}</span>
      </div>
    </div>
  );
}