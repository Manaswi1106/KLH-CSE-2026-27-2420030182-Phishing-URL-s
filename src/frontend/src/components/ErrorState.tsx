import React, { useState } from 'react';
import { AlertTriangle, RefreshCw, Terminal, Copy, Check, ChevronDown, ChevronUp, ExternalLink } from 'lucide-react';
import { getApiEndpoint } from '../services/api';

interface ErrorStateProps {
  onRetry: () => void;
  errorMessage?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  onRetry,
  errorMessage,
}) => {
  const [showTroubleshoot, setShowTroubleshoot] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);

  const corsSnippet = `# In your FastAPI backend main.py:
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["http://localhost:3000", "http://127.0.0.1:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)`;

  const handleCopyCode = () => {
    navigator.clipboard.writeText(corsSnippet);
    setCopiedCode(true);
    setTimeout(() => setCopiedCode(false), 2000);
  };

  const currentEndpoint = getApiEndpoint();

  return (
    <div className="max-w-3xl mx-auto px-4 py-8 animate-in fade-in duration-200">
      <div className="cyber-card rounded-2xl p-6 sm:p-8 border-rose-500/30 bg-[#120e1a]/80 shadow-[0_0_35px_rgba(244,63,94,0.15)]">
        <div className="flex items-start space-x-4">
          <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-rose-950/80 border border-rose-500/40 text-rose-400 shrink-0 shadow-lg shadow-rose-900/30">
            <AlertTriangle className="w-6 h-6" />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <span className="px-2 py-0.5 rounded text-[10px] uppercase font-mono font-bold bg-rose-950 text-rose-300 border border-rose-800">
                Connection Failed
              </span>
            </div>

            <h3 className="text-xl font-bold text-white font-display tracking-tight">
              Backend unavailable
            </h3>

            <p className="text-sm text-slate-300 mt-1.5 leading-relaxed">
              Make sure the PhishXpert FastAPI server is running at{' '}
              <code className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 font-mono-code text-cyan-300">
                {currentEndpoint}
              </code>
            </p>

            {errorMessage && (
              <div className="mt-3 p-3 rounded-lg bg-slate-950/80 border border-slate-800 text-xs font-mono-code text-rose-300 break-words">
                {errorMessage}
              </div>
            )}

            {/* Action Buttons */}
            <div className="mt-5 flex flex-wrap items-center gap-3">
              <button
                id="retry-analysis-btn"
                type="button"
                onClick={onRetry}
                className="px-4 py-2 bg-rose-600 hover:bg-rose-500 text-white font-medium text-xs rounded-xl flex items-center space-x-2 transition-all shadow-md shadow-rose-600/30 active:scale-95 cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Retry Connection</span>
              </button>

              <button
                type="button"
                onClick={() => setShowTroubleshoot(!showTroubleshoot)}
                className="px-3.5 py-2 bg-slate-900 hover:bg-slate-800 border border-slate-800 hover:border-slate-700 text-slate-300 text-xs rounded-xl flex items-center space-x-1.5 transition-colors cursor-pointer"
              >
                <Terminal className="w-3.5 h-3.5 text-slate-400" />
                <span>FastAPI / CORS Instructions</span>
                {showTroubleshoot ? (
                  <ChevronUp className="w-3.5 h-3.5 ml-1" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5 ml-1" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Collapsible Troubleshoot & CORS Guide */}
        {showTroubleshoot && (
          <div className="mt-6 pt-5 border-t border-slate-800/80 text-xs text-slate-300 space-y-4 animate-in slide-in-from-top-2 duration-200">
            <div className="space-y-2">
              <p className="font-semibold text-white flex items-center space-x-1.5">
                <span>1. Verify FastAPI server is running:</span>
              </p>
              <pre className="p-2.5 rounded-lg bg-slate-950 border border-slate-800 text-slate-300 font-mono-code overflow-x-auto text-[11px]">
                uvicorn main:app --reload --host 127.0.0.1 --port 8001
              </pre>
            </div>

            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <p className="font-semibold text-white">
                  2. Ensure FastAPI has CORS enabled:
                </p>
                <button
                  type="button"
                  onClick={handleCopyCode}
                  className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 text-[11px] font-mono-code flex items-center space-x-1"
                >
                  {copiedCode ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedCode ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <pre className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-cyan-300 font-mono-code overflow-x-auto text-[11px] leading-relaxed">
                {corsSnippet}
              </pre>
            </div>

            <div className="p-3 rounded-lg bg-cyan-950/30 border border-cyan-500/20 text-slate-300 text-xs leading-relaxed">
              <strong className="text-cyan-300">Browser Security Notice:</strong> If accessing this web app via HTTPS (such as the cloud preview), browsers prevent requests to plain <code className="font-mono-code text-cyan-200">http://127.0.0.1:8001</code> due to Mixed Content. Run the frontend locally using <code className="font-mono-code text-cyan-200">npm run dev</code> (runs on http://localhost:3000) or use a reverse tunnel.
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
