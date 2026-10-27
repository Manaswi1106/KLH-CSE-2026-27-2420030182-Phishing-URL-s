import React from 'react';
import {
  Layers,
  ShieldAlert,
  FileCode,
  FileText,
  Link2,
  ExternalLink,
  FormInput,
  KeyRound,
  LogIn,
  Percent,
  Share2
} from 'lucide-react';
import { WebpageEvidence } from '../types/api';

interface Props {
  evidence: WebpageEvidence;
}

export const EvidenceGrid: React.FC<Props> = ({ evidence }) => {
  const items = [
    ['Suspicious Keywords', evidence.suspicious_keyword_count, ShieldAlert],
    ['HTML Length', evidence.html_length, FileCode],
    ['Text Length', evidence.text_length, FileText],
    ['Links', evidence.num_links, Link2],
    ['External Links', evidence.external_links, ExternalLink],
    ['Forms', evidence.form_count, FormInput],
    ['Password Fields', evidence.password_field_count, KeyRound],
    ['Login Form', evidence.has_login_form ? 'Yes' : 'No', LogIn],
    ['External Forms', evidence.external_form_count, Share2],
    [
      'External Link Ratio',
      `${(evidence.external_link_ratio * 100).toFixed(2)}%`,
      Percent,
    ],
    [
      'External Form Ratio',
      `${(evidence.external_form_ratio * 100).toFixed(2)}%`,
      Percent,
    ],
  ];

  return (
    <div className="cyber-card rounded-2xl p-6">
      <div className="flex justify-between mb-6">
        <div className="flex gap-3">
          <div className="p-2 rounded-lg bg-cyan-950 border border-cyan-500/30">
            <Layers className="text-cyan-400" />
          </div>

          <div>
            <h3 className="text-lg font-bold text-white">
              23 Hybrid Security Features
            </h3>

            <p className="text-xs text-slate-400">
              Engineered URL/HTML features combined with DistilBERT semantic
              representations
            </p>
          </div>
        </div>

        <span className="px-3 py-1 rounded bg-slate-900 border border-slate-800 text-xs text-slate-300">
          23 Hybrid Security Features
        </span>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {items.map(([label, value, Icon]: any) => (
          <div key={label} className="p-4 rounded-xl bg-slate-950 border border-slate-800">
            <Icon className="w-4 h-4 text-cyan-400 mb-3" />

            <div className="text-xs text-slate-400 mb-1">{label}</div>

            <div className="text-xl font-bold text-white">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
};