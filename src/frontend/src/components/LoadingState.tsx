import React from 'react';
import { Shield, Radio, Activity, Search } from 'lucide-react';

interface LoadingStateProps {
  analyzedUrl?: string;
}

export const LoadingState: React.FC<LoadingStateProps> = ({ analyzedUrl }) => {
  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="cyber-card rounded-2xl p-8 sm:p-12 text-center relative overflow-hidden border-cyan-500/30 shadow-[0_0_40px_rgba(6,182,212,0.15)]">
        {/* Animated scanning radar beam */}
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-cyan-500/5 to-transparent animate-pulse pointer-events-none" />

        <div className="relative z-10 flex flex-col items-center justify-center">
          {/* Central Radar Pulse */}
          <div className="relative flex items-center justify-center w-24 h-24 mb-6">
            {/* Outer ripples */}
            <div className="absolute inset-0 rounded-full border border-cyan-500/20 animate-ping duration-1000" />
            <div className="absolute -inset-3 rounded-full border border-cyan-500/10 animate-pulse" />

            {/* Inner rotating ring */}
            <div className="w-20 h-20 rounded-full border-2 border-dashed border-cyan-400/40 animate-spin" style={{ animationDuration: '8s' }} />

            {/* Center icon */}
            <div className="absolute flex items-center justify-center w-14 h-14 rounded-full bg-slate-950 border border-cyan-500/50 shadow-[0_0_20px_rgba(6,182,212,0.4)]">
              <Shield className="w-7 h-7 text-cyan-400 animate-pulse" />
            </div>
          </div>

          {/* Heading */}
          <h2 className="text-2xl font-bold text-white tracking-tight font-display mb-2">
            Analyzing website...
          </h2>

          {/* Subtext */}
          <p className="text-sm text-cyan-300/80 font-medium mb-4">
            Extracting URL and webpage security signals
          </p>

          {analyzedUrl && (
            <div className="px-3.5 py-1.5 rounded-lg bg-slate-950/80 border border-slate-800 text-xs text-slate-300 font-mono-code max-w-md truncate mb-6">
              Target: <span className="text-cyan-300">{analyzedUrl}</span>
            </div>
          )}

          {/* Pipeline stage indicator badges */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-xl w-full text-xs font-mono-code">
            <div className="flex items-center space-x-2 p-2.5 rounded-lg bg-slate-900/60 border border-cyan-500/20 text-cyan-300">
              <Radio className="w-3.5 h-3.5 animate-spin text-cyan-400" />
              <span>URL Fetch</span>
            </div>
            <div className="flex items-center space-x-2 p-2.5 rounded-lg bg-slate-900/60 border border-cyan-500/20 text-cyan-300">
              <Search className="w-3.5 h-3.5 animate-pulse text-cyan-400" />
              <span>HTML Parsing</span>
            </div>
            <div className="flex items-center space-x-2 p-2.5 rounded-lg bg-slate-900/60 border border-cyan-500/20 text-cyan-300">
              <Activity className="w-3.5 h-3.5 animate-bounce text-cyan-400" />
              <span>Hybrid Model Analysis</span>
            </div>
            <div className="flex items-center space-x-2 p-2.5 rounded-lg bg-slate-900/60 border border-cyan-500/20 text-cyan-300">
              <Shield className="w-3.5 h-3.5 animate-pulse text-cyan-400" />
              <span>Verification</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
