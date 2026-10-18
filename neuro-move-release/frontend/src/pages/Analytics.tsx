import React, { useState } from 'react'

const metrics = [
  { name: 'MiniRocket', value: 98.63, trend: '+2.1%', trendDir: 'up' },
  { name: 'CNN-LSTM', value: 98.06, trend: '+1.5%', trendDir: 'up' },
  { name: 'CNN+Transformer', value: 96.50, trend: '-0.8%', trendDir: 'down' },
  { name: 'EEGNet', value: 95.20, trend: '+0.3%', trendDir: 'up' },
]

export function Analytics() {
  return (
    <div className="main-content">
      <div className="page-header">
        <h1 className="page-title">Analytics</h1>
        <p className="page-subtitle">Model performance metrics and training analytics</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-1.5rem">
        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Model Accuracy Comparison</h3>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              {metrics.map((m) => (
                <div key={m.name} className="flex items-center justify-between p-3 rounded-lg bg-white border border-slate-200">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-sky-50 flex items-center justify-center text-sky-600">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M12 2L2 7l10 5 10-5-10-5-10 5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
                    </div>
                    <span className="text-slate-700 font-medium">{m.name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-slate-900">{m.value}%</span>
                    <span className={`text-xs font-medium ${m.trendDir === 'up' ? 'text-emerald-600' : 'text-red-600'}`}>
                      {m.trend}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h3 className="card-title">Model Architecture Comparison</h3>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              {[
                { name: 'MiniRocket', desc: 'Deterministic PPV transform', score: 98.6 },
                { name: 'CNN-LSTM', desc: 'Hybrid spatial-temporal', score: 98.1 },
                { name: 'CNN+Transformer', desc: 'Multi-head attention', score: 96.5 },
                { name: 'EEGNet', desc: 'Depthwise separable', score: 95.2 },
              ].map((m) => (
                <div key={m.name} className="flex items-center gap-3">
                  <div className="w-20 h-2 bg-slate-200 rounded-full overflow-hidden">
                    <div className="h-full bg-sky-500" style={{ width: `${m.score}%` }}></div>
                  </div>
                  <div className="flex-1">
                    <div className="flex justify-between text-sm">
                      <span className="text-slate-700 font-medium">{m.name}</span>
                      <span className="text-slate-600 font-mono">{m.score}%</span>
                    </div>
                    <div className="text-xs text-slate-500">{m.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="card mt-4">
        <div className="card-header">
          <h3 className="card-title">Performance Metrics</h3>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Average Accuracy</div>
              <div className="text-2xl font-bold text-slate-900 font-mono">97.1%</div>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Avg Latency</div>
              <div className="text-2xl font-bold text-slate-900 font-mono">2.03 ms</div>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Models Trained</div>
              <div className="text-2xl font-bold text-slate-900 font-mono">127</div>
            </div>
            <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
              <div className="text-xs text-slate-500 uppercase tracking-wider mb-1">Avg Std Dev</div>
              <div className="text-2xl font-bold text-slate-900 font-mono">±1.45%</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}