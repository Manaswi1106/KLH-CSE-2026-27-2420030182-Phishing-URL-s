import React, { useState } from 'react';
import {
  ScanSearch,
  Globe,
  ArrowRight,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';

interface UrlAnalyzerProps {
  onAnalyze: (url: string) => void;
  isLoading: boolean;
  currentUrl?: string;
}

const DEMO_PRESETS = [
  'https://www.google.com',
  'https://www.youtube.com',
  'https://www.github.com',
  'https://www.wikipedia.org',
];

export const UrlAnalyzer: React.FC<UrlAnalyzerProps> = ({
  onAnalyze,
  isLoading,
  currentUrl = '',
}) => {
  const [urlInput, setUrlInput] = useState(currentUrl);
  const [validationError, setValidationError] = useState<string | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleanUrl = urlInput.trim();

    if (!cleanUrl) {
      setValidationError('Please enter a website URL to begin analysis.');
      return;
    }

    setValidationError(null);
    onAnalyze(cleanUrl);
  };

  const handleSelectPreset = (presetUrl: string) => {
    setUrlInput(presetUrl);
    setValidationError(null);
    onAnalyze(presetUrl);
  };

  return (
    <section className="relative pt-8 pb-12">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[650px] h-[350px] bg-cyan-500/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="max-w-4xl mx-auto text-center relative z-10 px-4">
        <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 rounded-full bg-cyan-950/60 border border-cyan-500/30 text-cyan-300 text-xs font-mono mb-6">
          <Sparkles className="w-3.5 h-3.5" />
          <span>Hybrid Transformer–ML Phishing Detection & Explainable AI</span>
        </div>

        <h1 className="text-5xl lg:text-6xl font-black text-white leading-tight">
          Detect Phishing.
          <br />
          <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
            Explain Every Decision.
          </span>
        </h1>

        <p className="text-slate-300 text-lg mt-6 max-w-3xl mx-auto leading-relaxed">
          Analyze website URLs using <b className="text-cyan-400">DistilBERT semantic embeddings</b>,{' '}
          <b className="text-indigo-400">DistilBERT + LightGBM</b>, <b className="text-cyan-400">Domain Trust Verification</b>,
          webpage evidence, and <b className="text-emerald-400">TreeSHAP Explainability</b>.
        </p>

        <div className="cyber-card rounded-2xl p-3 mt-10">
          <form onSubmit={handleSubmit} className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Globe className="absolute left-4 top-1/2 -translate-y-1/2 text-cyan-400 w-5 h-5" />
              <input
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                placeholder="Enter a website URL..."
                className="w-full pl-12 pr-4 py-3 rounded-xl bg-slate-950 border border-slate-800 text-white"
              />
            </div>

            <button className="px-6 py-3 rounded-xl bg-cyan-500 text-white font-semibold">
              {isLoading ? 'Analyzing...' : 'Analyze URL'}
            </button>
          </form>

          {validationError && (
            <div className="mt-3 text-rose-400 text-sm flex items-center gap-2">
              <ShieldAlert className="w-4 h-4" />
              {validationError}
            </div>
          )}
        </div>

        <div className="flex flex-wrap justify-center gap-2 mt-5">
          {DEMO_PRESETS.map((url) => (
            <button
              key={url}
              onClick={() => handleSelectPreset(url)}
              className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-xs text-slate-300 hover:border-cyan-500"
            >
              {url.replace('https://www.', '')}
            </button>
          ))}
        </div>
      </div>
    </section>
  );
};