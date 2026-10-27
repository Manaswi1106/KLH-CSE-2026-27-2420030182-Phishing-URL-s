import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { UrlAnalyzer } from './components/UrlAnalyzer';
import { ResultCard } from './components/ResultCard';
import { MLAnalysis } from './components/MLAnalysis';
import { DomainVerification } from './components/DomainVerification';
import { EvidenceGrid } from './components/EvidenceGrid';
import { ExplanationCard } from './components/ExplanationCard';
import { AnalysisHistory } from './components/AnalysisHistory';
import { LoadingState } from './components/LoadingState';
import { ErrorState } from './components/ErrorState';
import { EmptyState } from './components/EmptyState';
import { PredictionResponse, HistoryItem } from './types/api';
import { analyzeUrl, getApiEndpoint } from './services/api';
import ShapExplanation from "./components/ShapExplanation";

const HISTORY_STORAGE_KEY = 'phishxpert_session_history';

export default function App() {
  const [currentUrl, setCurrentUrl] = useState<string>('');
  const [analysisResult, setAnalysisResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [endpoint, setEndpoint] = useState<string>(getApiEndpoint());

  // Load session history from localStorage on initial render
  useEffect(() => {
    try {
      const saved = localStorage.getItem(HISTORY_STORAGE_KEY);
      if (saved) {
        const parsed: HistoryItem[] = JSON.parse(saved);
        if (Array.isArray(parsed)) {
          setHistory(parsed);
        }
      }
    } catch {
      // ignore parse error
    }
  }, []);

  // Save history to localStorage
  const saveHistory = (items: HistoryItem[]) => {
    setHistory(items);
    try {
      localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(items));
    } catch {
      // ignore storage error
    }
  };

  const handleAnalyze = async (urlToAnalyze: string) => {
    const cleanUrl = urlToAnalyze.trim();
    if (!cleanUrl) return;

    setCurrentUrl(cleanUrl);
    setIsLoading(true);
    setErrorMsg(null);

    try {
      const data = await analyzeUrl(cleanUrl, endpoint);
      setAnalysisResult(data);

      // Create new history entry
      const now = new Date();
      const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      const newItem: HistoryItem = {
        id: `${Date.now()}-${Math.random().toString(36).substring(2, 7)}`,
        url: cleanUrl,
        final_decision: data.final_decision,
        prediction: data.prediction,
        confidence: data.confidence,
        timestamp: timeString,
        result: data,
      };

      // Filter out duplicate identical URL if recent, add to top (max 10 items)
      const updatedHistory = [newItem, ...history.filter(h => h.url !== cleanUrl)].slice(0, 10);
      saveHistory(updatedHistory);
    } catch (err: unknown) {
      if (err && typeof err === 'object' && 'message' in err) {
        setErrorMsg((err as { message: string }).message);
      } else {
        setErrorMsg('Unable to connect to PhishXpert backend. Make sure FastAPI is running on port 8001.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleRetry = () => {
    if (currentUrl) {
      handleAnalyze(currentUrl);
    }
  };

  const handleSelectHistoryItem = (item: HistoryItem) => {
    setCurrentUrl(item.url);
    setAnalysisResult(item.result);
    setErrorMsg(null);
    window.scrollTo({ top: 400, behavior: 'smooth' });
  };

  const handleClearHistory = () => {
    setHistory([]);
    try {
      localStorage.removeItem(HISTORY_STORAGE_KEY);
    } catch {
      // ignore
    }
  };

  return (
    <div className="min-h-screen bg-[#070b14] text-slate-100 flex flex-col selection:bg-cyan-500/30 selection:text-cyan-200">
      {/* Background Cyber Grid */}
      <div className="fixed inset-0 cyber-grid opacity-70 pointer-events-none z-0" />

      {/* Top App Header */}
      <Header onEndpointChange={(newEp) => setEndpoint(newEp)} />

      {/* Main Content Area */}
      <main className="relative z-10 flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pb-16">
        {/* Hero & URL Input Card */}
        <UrlAnalyzer
          onAnalyze={handleAnalyze}
          isLoading={isLoading}
          currentUrl={currentUrl}
        />

        {/* Loading Inflight View */}
        {isLoading && <LoadingState analyzedUrl={currentUrl} />}

        {/* Error State View */}
        {!isLoading && errorMsg && (
          <ErrorState onRetry={handleRetry} errorMessage={errorMsg} />
        )}

        {/* Active Analysis Result Display */}
        {!isLoading && !errorMsg && analysisResult && (
          <div className="space-y-6 animate-in fade-in duration-300">
            {/* 1. Primary Final Decision Card */}
            <ResultCard data={analysisResult} url={currentUrl} />

            {/* 2. Side-by-side Dual Cards: ML Analysis & Domain Verification */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <MLAnalysis data={analysisResult} />
              <DomainVerification data={analysisResult.domain_verification} />
            </div>

            {/* 3. 11-point Evidence Signal Grid */}
              <EvidenceGrid evidence={analysisResult.evidence} />

            {/* 4. TreeSHAP Feature Importance */}
              <ShapExplanation features={analysisResult.shap_explanation || []} />

            {/* 5. Explainable AI Decision Reasoning */}
              <ExplanationCard data={analysisResult} />
          </div>
        )}

        {/* Empty State before any analysis has run */}
        {!isLoading && !errorMsg && !analysisResult && <EmptyState />}

        {/* Session Analysis History */}
        <AnalysisHistory
          history={history}
          onSelect={handleSelectHistoryItem}
          onClear={handleClearHistory}
          selectedUrl={analysisResult ? currentUrl : undefined}
        />
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-slate-800/80 bg-[#090e1c] py-6 text-xs text-slate-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <div className="flex items-center space-x-2">
            <span className="font-display font-semibold text-slate-400">PhishXpert</span>
            <span>—</span>
            <span>AI-Powered Detection of Phishing Websites Using Machine Learning and Explainable AI</span>
          </div>

          <div className="flex items-center space-x-4 font-mono-code text-[11px]">
            <span>FastAPI: <code className="text-cyan-400 font-semibold">{endpoint}</code></span>
            <span>•</span>
            <span>Client: Vite + React + TypeScript</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
