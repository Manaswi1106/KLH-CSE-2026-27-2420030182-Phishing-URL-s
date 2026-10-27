import React from 'react';
import { History, Clock, ArrowUpRight, Trash2, CheckCircle2, ShieldAlert, AlertTriangle } from 'lucide-react';
import { HistoryItem, PredictionResponse } from '../types/api';
import { formatPercent } from '../lib/utils';

interface AnalysisHistoryProps {
  history: HistoryItem[];
  onSelect: (item: HistoryItem) => void;
  onClear: () => void;
  selectedUrl?: string;
}

export const AnalysisHistory: React.FC<AnalysisHistoryProps> = ({
  history,
  onSelect,
  onClear,
  selectedUrl,
}) => {
  if (history.length === 0) {
    return null;
  }

  return (
    <div className="cyber-card rounded-2xl p-6 mt-8">
      <div className="flex items-center justify-between pb-4 border-b border-slate-800/80 mb-4">
        <div className="flex items-center space-x-2.5">
          <div className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-cyan-400">
            <History className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white tracking-tight font-display">
              Analysis History
            </h3>
            <p className="text-[11px] text-slate-400">
              Session cache ({history.length} analyzed targets)
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onClear}
          title="Clear session history"
          className="text-xs text-slate-500 hover:text-rose-400 font-mono-code flex items-center space-x-1 transition-colors px-2 py-1 rounded hover:bg-rose-950/30"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Clear</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {history.map((item) => {
          const isBenign = item.final_decision.toLowerCase() === 'benign';
          const isPhishing = item.final_decision.toLowerCase() === 'phishing';
          const isSelected = selectedUrl === item.url;

          const conf = typeof item.confidence === 'number'
            ? (item.confidence > 1 ? `${item.confidence.toFixed(1)}%` : formatPercent(item.confidence, false))
            : `${item.confidence}`;

          return (
            <button
              key={item.id}
              type="button"
              onClick={() => onSelect(item)}
              className={`p-3.5 rounded-xl border text-left transition-all group flex flex-col justify-between cursor-pointer ${
                isSelected
                  ? 'bg-slate-900 border-cyan-500/60 shadow-[0_0_15px_rgba(6,182,212,0.15)]'
                  : 'bg-slate-950/70 border-slate-800/90 hover:border-slate-700 hover:bg-slate-900/60'
              }`}
            >
              <div className="flex items-center justify-between mb-2 w-full">
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono-code font-bold uppercase flex items-center space-x-1 border ${
                  isBenign
                    ? 'bg-emerald-950/80 text-emerald-300 border-emerald-800'
                    : isPhishing
                    ? 'bg-rose-950/80 text-rose-300 border-rose-800'
                    : 'bg-amber-950/80 text-amber-300 border-amber-800'
                }`}>
                  {isBenign ? <CheckCircle2 className="w-3 h-3 shrink-0" /> : <ShieldAlert className="w-3 h-3 shrink-0" />}
                  <span>{item.final_decision}</span>
                </span>

                <div className="flex items-center space-x-1 text-[11px] text-slate-500 font-mono-code">
                  <Clock className="w-3 h-3" />
                  <span>{item.timestamp}</span>
                </div>
              </div>

              <div className="w-full">
                <div className="text-xs font-mono-code text-slate-200 truncate font-semibold group-hover:text-cyan-300 transition-colors" title={item.url}>
                  {item.url}
                </div>
                <div className="flex items-center justify-between text-[11px] text-slate-400 mt-1">
                  <span>ML: {item.prediction}</span>
                  <span className="font-mono-code text-cyan-400">{conf}</span>
                </div>
              </div>

              <div className="mt-2 pt-2 border-t border-slate-800/80 flex items-center justify-between text-[10px] text-slate-500 w-full group-hover:text-slate-300">
                <span>Click to restore</span>
                <ArrowUpRight className="w-3 h-3 group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform" />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};
