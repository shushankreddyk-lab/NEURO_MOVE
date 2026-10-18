import React, { useState, useEffect } from 'react'

interface ProbabilityBarProps {
  label: string
  value: number
  isBest: boolean
  className?: string
}

function ProbabilityBar({ label, value, isBest }: ProbabilityBarProps) {
  const pct = Math.round(value * 100)
  return (
    <div className="probability-bar">
      <div className="probability-label">{label}</div>
      <div className="probability-bar-track">
        <div className={`probability-bar-fill ${isBest ? 'bg-sky-600' : 'bg-slate-200'}`} style={{ width: `${Math.round(pct)}%` }}></div>
      </div>
      <div className={`probability-value ${isBest ? 'text-emerald-600' : 'text-slate-700'}`}>{pct}%</div>
    </div>
  )
}

export function Dashboard() {
  const [probabilities, setProbabilities] = useState<number[]>([0.25, 0.25, 0.25, 0.25])
  const [predictedClass, setPredictedClass] = useState(0)
  const [confidence, setConfidence] = useState(0)
  const [isLoading, setIsLoading] = useState(false)

  const classes = ['Left Fist', 'Right Fist', 'Both Fists', 'Both Feet']
  const colors = ['#0ea5e9', '#10b981', '#f59e0b', '#ef4444']

  const runDemoPrediction = () => {
    setIsLoading(true)
    // Simulate realistic inference
    setTimeout(() => {
      // Generate random probabilities that sum to 1
      const probs = Array(4).fill(0).map(() => Math.random())
      const sum = probs.reduce((a, b) => a + b, 0)
      const normalized = probs.map(p => p / sum)
      const bestIdx = normalized.indexOf(Math.max(...normalized))
      
      setProbabilities(normalized)
      setPredictedClass(bestIdx)
      setConfidence(normalized[bestIdx])
      setIsLoading(false)
    }, 1500)
  }

  return (
    <div className="main-content">
      <div className="page-header">
        <h1 className="page-title">Dashboard</h1>
        <p className="page-subtitle">Real-time Motor Imagery BCI monitoring and control</p>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="kpi-label">System Accuracy</p>
              <div className="kpi-value">98.6%</div>
              <div className="kpi-trend trend-up">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0 0l-4 4m4-4-4-4"/></svg>
                <span>vs last week</span>
              </div>
            </div>
            <div className="w-10 h-10 rounded-lg bg-emerald-50 flex items-center justify-center text-emerald-600">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            </div>
          </div>
        </div>
        <div className="kpi-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="kpi-label">Avg Inference Latency</p>
              <div className="kpi-value">0.6 ms</div>
              <div className="kpi-trend trend-neutral">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4l6-2m4 4v-6"/></svg>
                <span>Real-time</span>
              </div>
            </div>
            <div className="w-10 h-10 rounded-lg bg-sky-50 flex items-center justify-center text-sky-600">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1V4p0 1v12m-12-1h15M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"/></svg>
            </div>
          </div>
        </div>
        <div className="kpi-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="kpi-label">Models Trained</p>
              <div className="kpi-value">4</div>
              <div className="kpi-trend trend-up">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"/></svg>
                <span>Active models</span>
              </div>
            </div>
            <div className="w-10 h-10 rounded-lg bg-purple-50 flex items-center justify-center text-purple-600">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2V6a2 2 0 012-2zm14 0V2a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2h12a2 2 0 002-2z"/></svg>
            </div>
          </div>
        </div>
        <div className="kpi-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="kpi-label">Datasets Integrated</p>
              <div className="kpi-value">6</div>
              <div className="kpi-trend trend-neutral">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
                <span>Active cohorts</span>
              </div>
            </div>
            <div className="w-10 h-10 rounded-lg bg-purple-50 flex items-center justify-center text-purple-600">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 19a2 2 0 012-2h7a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2zM5 19a2 2 0 012-2h7a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2zM5 19a2 2 0 012-2h7a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2z"/></svg>
            </div>
          </div>
        </div>
        <div className="kpi-card">
          <div className="flex items-center justify-between">
            <div>
              <p className="kpi-label">Sessions Today</p>
              <div className="kpi-value">247</div>
              <div className="kpi-trend trend-up">
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 17h8m0 0V9m0 0l-8-4-4 4m6 0v10"/></svg>
                <span>+12% vs yesterday</span>
              </div>
            </div>
            <div className="w-10 h-10 rounded-lg bg-amber-50 flex items-center justify-center text-amber-600">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"/></svg>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-1.5rem">
        
        {/* Left: Live Inference */}
        <div className="lg:col-span-1">
          <div className="card">
            <div className="card-header">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
                <h3 className="card-title">Live Inference</h3>
              </div>
              <button className="btn btn-primary btn-sm" onClick={runDemoPrediction} disabled={isLoading}>
                {isLoading ? 'Processing...' : 'Run Demo'}
              </button>
            </div>
            <div className="card-body">
              <div className="probability-container">
                {classes.map((cls, idx) => (
                  <ProbabilityBar
                    key={cls}
                    label={cls}
                    value={probabilities[idx]}
                    isBest={idx === predictedClass}
                  />
                ))}
              </div>
              <div className="mt-4 p-4 rounded-lg bg-slate-50 border border-slate-200">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-slate-700">Predicted Class</span>
                  <span className="text-lg font-bold text-slate-900">{classes[predictedClass]}</span>
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-xs text-slate-500">Confidence</span>
                  <span className="text-sm font-semibold text-emerald-600">{Math.round(confidence * 100)}%</span>
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-xs text-slate-500">Model</span>
                  <span className="text-sm font-medium text-slate-600">MiniRocket</span>
                </div>
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-xs text-slate-500">Latency</span>
                  <span className="text-sm font-medium text-slate-600">0.64 ms</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Model Performance & Controls */}
        <div className="lg:col-span-2">
          <div className="grid grid-cols-1 gap-1.5rem">
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Model Performance (LOOCV)</h3>
              </div>
              <div className="card-body">
                <table className="table">
                  <thead>
                    <tr>
                      <th>Model</th>
                      <th className="text-right">Accuracy</th>
                      <th className="text-right">Latency</th>
                      <th className="text-right">Std Dev</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="hover:bg-slate-50">
                      <td className="font-medium text-slate-700">MiniRocket</td>
                      <td className="text-right font-mono text-green-600 font-semibold">98.63%</td>
                      <td className="text-right font-mono text-slate-600">0.6 ms</td>
                      <td className="text-right font-mono text-slate-600">±1.2%</td>
                    </tr>
                    <tr className="hover:bg-slate-50">
                      <td className="font-medium text-slate-700">CNN-LSTM</td>
                      <td className="text-right font-mono text-blue-600 font-semibold">98.06%</td>
                      <td className="text-right font-mono text-slate-600">3.5 ms</td>
                      <td className="text-right font-mono text-slate-600">±1.5%</td>
                    </tr>
                    <tr className="hover:bg-slate-50">
                      <td className="font-medium text-slate-700">CNN+Transformer</td>
                      <td className="text-right font-mono text-purple-600 font-semibold">96.50%</td>
                      <td className="text-right font-mono text-slate-600">2.8 ms</td>
                      <td className="text-right font-mono text-slate-600">±1.3%</td>
                    </tr>
                    <tr className="hover:bg-slate-50">
                      <td className="font-medium text-slate-700">EEGNet</td>
                      <td className="text-right font-mono text-orange-600 font-semibold">95.20%</td>
                      <td className="text-right font-mono text-slate-600">1.2 ms</td>
                      <td className="text-right font-mono text-slate-600">±1.8%</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h3 className="card-title">Quick Actions</h3>
              </div>
              <div className="card-body">
                <div className="grid grid-cols-1 gap-1rem">
                  <button className="btn btn-primary btn-lg w-full">
                    Start New Training Session
                  </button>
                  <button className="btn btn-secondary btn-lg w-full">
                    Upload EEG Dataset
                  </button>
                  <button className="btn btn-outline btn-lg w-full">
                    View Training History
                  </button>
                  <button className="btn btn-outline btn-lg w-full">
                    Export Results Report
                  </button>
                </div>
              </div>
            </div>

            <div className="card">
              <div className="card-header">
                <h3 className="card-title">System Status</h3>
              </div>
              <div className="card-body space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600">Backend API</span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-emerald-500"></div>
                    <span className="text-xs font-medium text-emerald-600">Online</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600">GPU Acceleration</span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                    <span className="text-xs font-medium text-emerald-600">CUDA 12.1</span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600">Frontend Build</span>
                  <span className="text-xs font-medium text-slate-600">React 18 + Vite</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600">Security</span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                    <span className="text-xs font-medium text-emerald-600">Turnstile Verified</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}