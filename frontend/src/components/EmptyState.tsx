import React from 'react';
import { ShieldCheck, Cpu, Database, Microscope, Search, ArrowUp } from 'lucide-react';

export const EmptyState: React.FC = () => {
  return (
    <div className="max-w-3xl mx-auto px-4 py-12 text-center animate-in fade-in duration-300">
      <div className="cyber-card rounded-2xl p-8 sm:p-12 relative overflow-hidden border-slate-800">
        {/* Glow */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-cyan-500/5 blur-3xl pointer-events-none rounded-full" />

        {/* Shield Icon Graphic */}
        <div className="relative inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-slate-900/90 border border-cyan-500/30 text-cyan-400 mb-6 shadow-[0_0_30px_rgba(6,182,212,0.15)]">
          <ShieldCheck className="w-10 h-10" />
          <div className="absolute -bottom-2 -right-2 p-1.5 rounded-lg bg-indigo-950 border border-indigo-500/40 text-indigo-400">
            <Cpu className="w-4 h-4" />
          </div>
        </div>

        {/* Text */}
        <h3 className="text-2xl font-bold text-white tracking-tight font-display mb-2">
          Enter a URL above to begin security analysis.
        </h3>

        <p className="text-sm text-slate-400 max-w-md mx-auto leading-relaxed mb-8">
          PhishXpert will perform real-time URL lexical inspection, DOM webpage evidence extraction, DistilBERT + LightGBM analysis, and domain trust verification.
        </p>

        {/* Feature Pill Highlights */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 max-w-xl mx-auto text-left text-xs">
          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <div className="text-cyan-400 font-semibold mb-1 flex items-center space-x-1.5">
              <Microscope className="w-3.5 h-3.5" />
              <span>23 Hybrid Security Features</span>
            </div>
            <p className="text-slate-400 text-[11px] leading-snug">
              Audits forms, external link ratios, keyword flags, and script DOM sizes.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <div className="text-indigo-400 font-semibold mb-1 flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5" />
              <span>DistilBERT + LightGBM</span>
            </div>
            <p className="text-slate-400 text-[11px] leading-snug">
              Hybrid Transformer-ML architecture trained on adversarial phishing topologies.
            </p>
          </div>

          <div className="p-3.5 rounded-xl bg-slate-950/60 border border-slate-800/80">
            <div className="text-emerald-400 font-semibold mb-1 flex items-center space-x-1.5">
              <ShieldCheck className="w-3.5 h-3.5" />
              <span>Domain Trust Verification</span>
            </div>
            <p className="text-slate-400 text-[11px] leading-snug">
              Independent cryptographic and registration checks to prevent false positives.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
