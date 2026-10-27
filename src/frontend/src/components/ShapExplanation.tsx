import React from "react";
import { BarChart3, ArrowUpRight, ArrowDownRight } from "lucide-react";
import { ShapFeature } from "../types/api";

interface Props {
  features?: ShapFeature[];
}

export default function ShapExplanation({ features = [] }: Props) {
  return (
    <div className="cyber-card rounded-2xl p-6">
      <div className="flex items-center gap-3 mb-5">
        <div className="p-2 rounded-lg bg-violet-950 border border-violet-500/30">
          <BarChart3 className="w-5 h-5 text-violet-400" />
        </div>

        <div>
          <h3 className="text-lg font-bold text-white">
            TreeSHAP Explainability
          </h3>
          <p className="text-xs text-slate-400">
            Feature contribution to the Hybrid Prediction
          </p>
        </div>
      </div>

      {features.length === 0 ? (
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-5 text-center">
          <p className="text-slate-400 text-sm">
            TreeSHAP Explainability will appear after the backend returns feature
            importance values.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {features.map((item, i) => {
            const positive = item.impact >= 0;
            const width = Math.min(Math.abs(item.impact) * 100, 100);

            return (
              <div key={i}>
                <div className="flex justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {positive ? (
                      <ArrowUpRight className="w-4 h-4 text-rose-400" />
                    ) : (
                      <ArrowDownRight className="w-4 h-4 text-emerald-400" />
                    )}

                    <span className="text-sm text-white">
                      {item.feature}
                    </span>
                  </div>

                  <span
                    className={`text-sm font-mono ${
                      positive ? "text-rose-400" : "text-emerald-400"
                    }`}
                  >
                    {item.impact.toFixed(3)}
                  </span>
                </div>

                <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      positive
                        ? "bg-gradient-to-r from-rose-500 to-orange-400"
                        : "bg-gradient-to-r from-emerald-500 to-teal-400"
                    }`}
                    style={{ width: `${width}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}