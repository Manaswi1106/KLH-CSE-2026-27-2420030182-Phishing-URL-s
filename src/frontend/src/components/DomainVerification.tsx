import React from 'react';
import {
  ShieldCheck,
  ShieldAlert,
  Globe,
  Server,
  CheckCircle2,
  XCircle,
  Info,
} from 'lucide-react';
import { DomainVerification as DomainType } from '../types/api';

interface Props {
  data: DomainType;
}

export const DomainVerification: React.FC<Props> = ({ data }) => {
  const trusted = data.trusted;

  return (
    <div className="cyber-card rounded-2xl p-6">
      <div className="flex justify-between mb-5">
        <div className="flex gap-3">
          <div
            className={`p-2 rounded-lg border ${
              trusted
                ? 'bg-emerald-950 border-emerald-500/40'
                : 'bg-slate-900 border-slate-700'
            }`}
          >
            {trusted ? (
              <ShieldCheck className="text-emerald-400" />
            ) : (
              <ShieldAlert className="text-slate-400" />
            )}
          </div>

          <div>
            <h3 className="text-lg font-bold text-white">
              Domain Trust Verification
            </h3>

            <p className="text-xs text-slate-400">
              Independent registered-domain validation & trust authority verification
            </p>
          </div>
        </div>

        <span
          className={`px-3 py-1 rounded text-xs border ${
            trusted
              ? 'bg-emerald-950 border-emerald-500 text-emerald-300'
              : 'bg-slate-900 border-slate-700 text-slate-300'
          }`}
        >
          {trusted ? 'Trusted' : 'Unknown'}
        </span>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-5">
        <InfoBox icon={Globe} title="Hostname" value={data.hostname} />

        <InfoBox
          icon={Server}
          title="Registered Domain"
          value={data.registered_domain}
        />

        <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
          <div className="text-xs text-slate-400 mb-2">Trusted Authority</div>

          <div className="flex items-center gap-2">
            {trusted ? (
              <>
                <CheckCircle2 className="text-emerald-400 w-4 h-4" />
                <span className="text-emerald-400 font-semibold">Yes</span>
              </>
            ) : (
              <>
                <XCircle className="text-slate-400 w-4 h-4" />
                <span className="text-slate-300 font-semibold">No</span>
              </>
            )}
          </div>
        </div>
      </div>

      <div className="p-3 rounded-xl bg-slate-950 border border-slate-800 flex gap-2 text-xs text-slate-400">
        <Info className="w-4 h-4 text-cyan-400 mt-0.5" />

        <span>
          Domain verification is an independent security signal and is combined
          with the Hybrid Prediction during Hybrid Transformer-ML analysis.
        </span>
      </div>
    </div>
  );
};

const InfoBox = ({ icon: Icon, title, value }: any) => (
  <div className="p-3 rounded-xl bg-slate-950 border border-slate-800">
    <div className="flex items-center gap-1 text-xs text-slate-400 mb-2">
      <Icon className="w-3 h-3 text-cyan-400" />
      {title}
    </div>

    <div className="text-sm font-semibold text-white break-all">{value}</div>
  </div>
);