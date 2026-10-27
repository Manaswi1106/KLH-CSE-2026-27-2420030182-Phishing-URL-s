import React from 'react';
import {
  BrainCircuit,
  AlertTriangle,
  ShieldCheck,
  Cpu,
} from 'lucide-react';
import { PredictionResponse } from '../types/api';

interface Props {
  data: PredictionResponse;
}

export const MLAnalysis: React.FC<Props> = ({ data }) => {
  const phishing = data.prediction === 'Phishing';
  const conf = Number(data.confidence);

  return (
    <div className="cyber-card rounded-2xl p-6">
      <div className="flex justify-between mb-5">
        <div className="flex gap-3">
          <div className="p-2 rounded-lg bg-indigo-950 border border-indigo-500/30">
            <BrainCircuit className="text-indigo-400 w-5 h-5" />
          </div>

          <div>
            <h3 className="font-bold text-white text-lg">
              Hybrid Model Analysis
            </h3>

            <p className="text-xs text-slate-400">
              DistilBERT + LightGBM classification
            </p>
          </div>
        </div>

        <span className="text-xs px-2 py-1 rounded bg-slate-900 border border-slate-800 text-slate-300">
          DistilBERT + LightGBM
        </span>
      </div>

      <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 mb-5 flex justify-between">
        <span className="flex gap-2 text-xs text-slate-400">
          <Cpu className="w-4 h-4 text-cyan-400" />
          Model Architecture
        </span>

        <span className="text-xs text-white font-mono">
          DistilBERT + LightGBM
        </span>
      </div>

      <div className="flex justify-between items-end">
        <div>
          <p className="text-xs text-slate-400 uppercase">Hybrid Prediction</p>

          <div className="flex gap-2 mt-1">
            {phishing ? (
              <AlertTriangle className="text-rose-400" />
            ) : (
              <ShieldCheck className="text-emerald-400" />
            )}

            <span
              className={`text-2xl font-bold ${
                phishing ? 'text-rose-400' : 'text-emerald-400'
              }`}
            >
              {data.prediction}
            </span>
          </div>
        </div>

        <div className="text-right">
          <p className="text-xs text-slate-400 uppercase">
            Prediction Confidence
          </p>

          <div className="text-2xl text-cyan-300 font-bold">
            {conf.toFixed(2)}%
          </div>
        </div>
      </div>

      <div className="mt-4 h-3 bg-slate-900 rounded-full overflow-hidden">
        <div
          className={`h-full ${
            phishing
              ? 'bg-gradient-to-r from-amber-500 to-rose-500'
              : 'bg-gradient-to-r from-teal-500 to-emerald-500'
          }`}
          style={{ width: `${conf}%` }}
        />
      </div>
    </div>
  );
};