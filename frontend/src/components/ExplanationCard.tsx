import React from 'react';
import {
  Sparkles,
  BrainCircuit,
  ShieldCheck,
  ShieldAlert,
  CheckCircle2,
  ArrowRight,
  GitMerge,
  Info,
  Scale
} from 'lucide-react';
import { PredictionResponse } from '../types/api';
import { formatPercent } from '../lib/utils';

interface ExplanationCardProps {
  data: PredictionResponse;
}

export const ExplanationCard: React.FC<ExplanationCardProps> = ({ data }) => {
  const isConflict =
    data.prediction.toLowerCase() !== data.final_decision.toLowerCase();

  const confidenceFormatted =
    typeof data.confidence === 'number'
      ? data.confidence > 1
        ? `${data.confidence.toFixed(2)}%`
        : formatPercent(data.confidence, false)
      : `${data.confidence}`;

  return (
    <div className="cyber-card rounded-2xl p-6 sm:p-8 relative overflow-hidden">
      {/* Background Glow */}
      <div className="absolute top-0 right-1/4 w-72 h-72 bg-indigo-500/5 blur-3xl pointer-events-none" />

      {/* Header */}
      <div className="flex items-center space-x-3 mb-6">
        <div className="p-2.5 rounded-xl bg-gradient-to-br from-cyan-500/20 to-indigo-500/20 border border-cyan-500/30 text-cyan-400 shadow-md">
          <Sparkles className="w-6 h-6" />
        </div>

        <div>
          <h3 className="text-xl font-bold text-white tracking-tight font-display">
            Explainable AI Decision Pipeline
          </h3>

          <p className="text-xs text-slate-400">
            Hybrid Transformer-ML Architecture • Explainable Decision Reasoning
          </p>
        </div>
      </div>

      {/* Architecture */}
      <div className="mb-6 p-4 rounded-xl bg-slate-950/80 border border-slate-800">
        <div className="flex items-center justify-between mb-3 text-xs font-mono-code text-slate-400">
          <span className="flex items-center space-x-1.5">
            <Scale className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-semibold text-slate-300">
              Hybrid Transformer-ML Architecture
            </span>
          </span>

          {isConflict && (
            <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-amber-950 text-amber-300 border border-amber-800 animate-pulse">
              Domain Trust Override
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {/* STEP 1 */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono-code text-slate-400 uppercase">
                1. DistilBERT + LightGBM
              </span>
              <BrainCircuit className="w-4 h-4 text-indigo-400" />
            </div>

            <div>
              <div className="text-lg font-bold font-display">
                <span
                  className={
                    data.prediction.toLowerCase() === 'phishing'
                      ? 'text-rose-400'
                      : 'text-emerald-400'
                  }
                >
                  {data.prediction}
                </span>
              </div>

              <div className="text-xs text-cyan-300 font-mono-code mt-1">
                Confidence: {confidenceFormatted}
              </div>
            </div>

            <div className="text-[11px] text-slate-400 mt-3 pt-2 border-t border-slate-800">
              DistilBERT semantic embeddings combined with 23 engineered URL
              and HTML security features.
            </div>
          </div>

          {/* STEP 2 */}
          <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 flex flex-col justify-between">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono-code text-slate-400 uppercase">
                2. Domain Trust Verification
              </span>

              <ShieldCheck className="w-4 h-4 text-cyan-400" />
            </div>

            <div>
              <div className="text-lg font-bold font-display">
                <span
                  className={
                    data.domain_verification.trusted
                      ? 'text-emerald-400'
                      : 'text-slate-300'
                  }
                >
                  {data.domain_verification.trusted
                    ? 'Trusted Domain'
                    : 'Unknown / Untrusted'}
                </span>
              </div>

              <div
                className="text-xs text-slate-300 font-mono-code mt-1 truncate"
                title={data.domain_verification.registered_domain}
              >
                {data.domain_verification.registered_domain}
              </div>
            </div>

            <div className="text-[11px] text-slate-400 mt-3 pt-2 border-t border-slate-800">
              Independent registered-domain validation used as an additional
              security signal.
            </div>
          </div>

          {/* STEP 3 */}
          <div
            className={`p-4 rounded-xl border flex flex-col justify-between ${
              data.final_decision.toLowerCase() === 'benign'
                ? 'border-emerald-500/40 bg-emerald-950/20'
                : data.final_decision.toLowerCase() === 'phishing'
                ? 'border-rose-500/40 bg-rose-950/20'
                : 'border-amber-500/40 bg-amber-950/20'
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-mono-code text-slate-400 uppercase">
                3. Hybrid Prediction
              </span>

              <GitMerge className="w-4 h-4 text-cyan-400" />
            </div>

            <div>
              <div
                className={`text-lg font-bold uppercase ${
                  data.final_decision.toLowerCase() === 'benign'
                    ? 'text-emerald-400'
                    : data.final_decision.toLowerCase() === 'phishing'
                    ? 'text-rose-400'
                    : 'text-amber-400'
                }`}
              >
                {data.final_decision}
              </div>

              <div className="text-xs text-slate-300 mt-1">
                Hybrid Prediction
              </div>
            </div>

            <div className="text-[11px] text-slate-400 mt-3 pt-2 border-t border-slate-800">
              Final phishing verdict generated by combining transformer
              semantics, LightGBM prediction, webpage evidence, and domain trust
              verification.
            </div>
          </div>
        </div>
      </div>

      {/* Decision Reason */}
      <div className="space-y-4">
        {data.decision_reason && (
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/90">
            <div className="flex items-center space-x-2 text-xs font-mono-code text-cyan-300 font-semibold mb-2 uppercase">
              <Info className="w-4 h-4 text-cyan-400" />
              <span>Decision Reason</span>
            </div>

            <p className="text-sm text-slate-200 leading-relaxed">
              {data.decision_reason}
            </p>
          </div>
        )}

        {data.explanation && (
          <div className="p-4 rounded-xl bg-slate-950/70 border border-slate-800/90">
            <div className="flex items-center space-x-2 text-xs font-mono-code text-indigo-300 font-semibold mb-2 uppercase">
              <Sparkles className="w-4 h-4 text-indigo-400" />
              <span>Hybrid AI Reasoning</span>
            </div>

            <p className="text-sm text-slate-300 leading-relaxed">
              {data.explanation}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};