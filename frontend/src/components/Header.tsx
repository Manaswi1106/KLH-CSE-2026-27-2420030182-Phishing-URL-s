import React, { useState } from 'react';
import { ShieldCheck, Cpu, Radio, Settings2, Check, RotateCcw } from 'lucide-react';
import { getApiEndpoint, setApiEndpoint, resetApiEndpoint } from '../services/api';

interface HeaderProps {
  onEndpointChange?: (newEndpoint: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ onEndpointChange }) => {
  const [showConfig, setShowConfig] = useState(false);
  const [endpointInput, setEndpointInput] = useState(getApiEndpoint());
  const [savedSuccess, setSavedSuccess] = useState(false);

  const handleSaveEndpoint = (e: React.FormEvent) => {
    e.preventDefault();
    setApiEndpoint(endpointInput);
    if (onEndpointChange) {
      onEndpointChange(endpointInput);
    }
    setSavedSuccess(true);
    setTimeout(() => setSavedSuccess(false), 2000);
  };

  const handleReset = () => {
    resetApiEndpoint();
    const def = getApiEndpoint();
    setEndpointInput(def);
    if (onEndpointChange) {
      onEndpointChange(def);
    }
  };

  return (
    <header className="border-b border-cyan-500/15 bg-[#0a1020]/90 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
        {/* Left: Branding */}
        <div className="flex items-center space-x-3.5">
          <div className="relative flex items-center justify-center w-11 h-11 rounded-xl bg-gradient-to-br from-cyan-500/20 via-blue-600/20 to-indigo-600/30 border border-cyan-500/40 shadow-[0_0_15px_rgba(6,182,212,0.25)]">
            <ShieldCheck className="w-6 h-6 text-cyan-400" />
            <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-cyan-400 rounded-full animate-ping opacity-75" />
            <div className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-cyan-400 rounded-full" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-xl font-bold tracking-tight text-white font-display">
                Phish<span className="text-cyan-400">Xpert</span>
              </span>
              <span className="px-2 py-0.5 text-[10px] uppercase font-mono tracking-wider font-semibold rounded bg-cyan-950/80 text-cyan-300 border border-cyan-500/30">
                v2.4 Core
              </span>
            </div>
            <p className="text-xs text-slate-400 font-medium tracking-normal hidden sm:block">
              Knowledge-Grounded Explainable Phishing Detection
            </p>
          </div>
        </div>

        {/* Right: Status Badges & Connection Config */}
        <div className="flex items-center space-x-2.5 sm:space-x-4">
          <div className="hidden md:flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-xs text-slate-300 font-mono-code">
            <Cpu className="w-3.5 h-3.5 text-indigo-400" />
            <span>AI Security Engine</span>
          </div>

          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-cyan-950/40 border border-cyan-500/30 text-xs text-cyan-300">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-400" />
            </span>
            <span className="font-medium tracking-wide">Live Analysis</span>
          </div>

          {/* Config Settings Trigger */}
          <button
            onClick={() => setShowConfig(!showConfig)}
            title="Configure FastAPI endpoint"
            aria-label="Configure FastAPI endpoint"
            className="p-2 rounded-lg bg-slate-900/90 border border-slate-800 text-slate-400 hover:text-cyan-400 hover:border-cyan-500/40 transition-colors"
          >
            <Settings2 className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Backend API Configuration Drawer / Dropdown */}
      {showConfig && (
        <div className="border-t border-slate-800 bg-[#0c1427] px-4 py-3.5 shadow-2xl animate-in slide-in-from-top-2 duration-200">
          <div className="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-3 text-xs">
            <div className="flex items-center space-x-2 text-slate-300">
              <Radio className="w-4 h-4 text-cyan-400 shrink-0" />
              <span>
                Backend API target: <code className="text-cyan-300 font-mono-code bg-slate-900 px-1.5 py-0.5 rounded border border-slate-800">{getApiEndpoint()}</code>
              </span>
            </div>

            <form onSubmit={handleSaveEndpoint} className="flex items-center gap-2 flex-wrap">
              <input
                type="text"
                value={endpointInput}
                onChange={(e) => setEndpointInput(e.target.value)}
                placeholder="http://127.0.0.1:8001/predict"
                className="px-3 py-1.5 bg-slate-950 border border-slate-700 rounded text-slate-200 font-mono-code text-xs w-64 focus:outline-none focus:border-cyan-500"
              />
              <button
                type="submit"
                className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded font-medium flex items-center space-x-1 transition-colors"
              >
                {savedSuccess ? <Check className="w-3.5 h-3.5" /> : null}
                <span>{savedSuccess ? 'Saved' : 'Update URL'}</span>
              </button>
              <button
                type="button"
                onClick={handleReset}
                title="Reset to default 127.0.0.1:8001"
                className="px-2 py-1.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded flex items-center space-x-1"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset</span>
              </button>
            </form>
          </div>
        </div>
      )}
    </header>
  );
};
