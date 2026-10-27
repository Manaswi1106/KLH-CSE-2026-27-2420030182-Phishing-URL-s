import React from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  AlertTriangle,
  Cpu,
  Globe2,
  CheckCircle2,
  Info
} from 'lucide-react';
import { PredictionResponse } from '../types/api';
import { formatPercent } from '../lib/utils';

interface ResultCardProps {
  data: PredictionResponse;
  url: string;
}

export const ResultCard: React.FC<ResultCardProps> = ({ data, url }) => {
  const decision = data.final_decision;
  const isBenign = decision.toLowerCase() === 'benign';
  const isPhishing = decision.toLowerCase() === 'phishing';

  const confidenceFormatted =
    typeof data.confidence === 'number'
      ? data.confidence > 1
        ? `${data.confidence.toFixed(2)}%`
        : formatPercent(data.confidence, false)
      : `${data.confidence}`;

  const theme = isBenign
    ? {
        border: 'border-emerald-500/40',
        bg: 'from-emerald-950/40 via-[#0a1520] to-[#070b14]',
        badge: 'bg-emerald-950 border-emerald-500 text-emerald-300',
        icon: ShieldCheck,
        text: 'text-emerald-400'
      }
    : isPhishing
    ? {
        border: 'border-rose-500/40',
        bg: 'from-rose-950/40 via-[#180d19] to-[#070b14]',
        badge: 'bg-rose-950 border-rose-500 text-rose-300',
        icon: ShieldAlert,
        text: 'text-rose-400'
      }
    : {
        border: 'border-amber-500/40',
        bg: 'from-amber-950/40 via-[#18150d] to-[#070b14]',
        badge: 'bg-amber-950 border-amber-500 text-amber-300',
        icon: AlertTriangle,
        text: 'text-amber-400'
      };

  const Icon = theme.icon;

  return (
    <div className={`cyber-card rounded-2xl p-6 bg-gradient-to-b ${theme.bg} ${theme.border}`}>
      <div className="flex justify-between border-b border-slate-800 pb-5">
        <div className="flex items-center gap-2">
          <Globe2 className="w-4 h-4 text-cyan-400" />
          <span className="text-xs text-slate-400 uppercase">Analyzed Target</span>
          <span className="font-mono text-white">{url}</span>
        </div>

        <div className="px-3 py-1 rounded bg-slate-900 border border-slate-800 text-xs text-slate-300 flex items-center gap-1">
          <Cpu className="w-3 h-3 text-indigo-400" />
          {data.model || 'PhishXpert Hybrid (DistilBERT + LightGBM)'}
        </div>
      </div>

      <div className="py-7 flex gap-5">
        <div className="w-20 h-20 rounded-2xl flex items-center justify-center bg-slate-900 border border-slate-800">
          <Icon className={`w-10 h-10 ${theme.text}`} />
        </div>

        <div>
          <div className="flex gap-2 items-center mb-2">
            <span className="text-xs uppercase tracking-widest text-slate-400">
              Final Decision
            </span>

            <span className={`px-2 py-1 rounded text-[10px] border ${theme.badge}`}>
              {data.final_decision}
            </span>
          </div>

          <h2 className={`text-4xl font-black ${theme.text}`}>
            {data.final_decision.toUpperCase()}
          </h2>

          <p className="mt-2 text-slate-300 max-w-2xl">
            {data.decision_reason}
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 border-t border-slate-800 pt-6">
        <Metric
          title="HYBRID PREDICTION"
          value={data.prediction}
          color={isPhishing ? 'text-rose-400' : 'text-emerald-400'}
        />

        <Metric
          title="PREDICTION CONFIDENCE"
          value={confidenceFormatted}
          color="text-cyan-300"
        />

        <Metric
          title="DOMAIN STATUS"
          value={data.domain_verification.status}
          icon={data.domain_verification.trusted ? CheckCircle2 : Info}
        />

        <Metric title="WEBSITE ACCESS" value={data.access_status} />

        <Metric
          title="ANALYSIS MODE"
          value={data.feature_mode || 'URL + HTML Features'}
        />
      </div>
    </div>
  );
};

const Metric = ({
  title,
  value,
  color = 'text-white',
  icon: Icon,
}: any) => (
  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
    <div className="text-[11px] text-slate-400 uppercase mb-1">{title}</div>
    <div className={`font-bold flex items-center gap-1 ${color}`}>
      {Icon && <Icon className="w-4 h-4" />}
      {value}
    </div>
  </div>
);