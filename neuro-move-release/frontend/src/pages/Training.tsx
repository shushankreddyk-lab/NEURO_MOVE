import React, { useState } from 'react'

interface ModelConfig {
  numKernels: number
  hidden: number
  dropout: number
  epochs: number
  batchSize: number
  learningRate: number
  subjects: string
  headEpochs: number
}

const defaultConfig: ModelConfig = {
  numKernels: 10000,
  hidden: 512,
  dropout: 0.15,
  epochs: 30,
  batchSize: 256,
  learningRate: 5e-4,
  subjects: '1-10',
  headEpochs: 30,
}

export function Training() {
  const [config, setConfig] = useState<ModelConfig>(defaultConfig)
  const [isTraining, setIsTraining] = useState(false)
  const [progress, setProgress] = useState(0)
  const [logs, setLogs] = useState<string[]>([])
  const [status, setStatus] = useState<'idle' | 'training' | 'completed'>('idle')

  const addLog = (msg: string) => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [...prev, `[${timestamp}] ${msg}`])
  }

  const startTraining = () => {
    setIsTraining(true)
    setStatus('training')
    setProgress(0)
    setLogs([])
    addLog('Initializing MiniRocket pipeline...')
    addLog(`Configuration: ${config.numKernels} kernels, ${config.epochs} epochs, batch ${config.batchSize}`)
    addLog(`Subjects: ${config.subjects}`)

    let currentProgress = 0
    const interval = setInterval(() => {
      currentProgress += Math.random() * 15
      if (currentProgress >= 100) {
        currentProgress = 100
        setProgress(100)
        setStatus('completed')
        setIsTraining(false)
        addLog('Training completed successfully!')
        addLog('Final accuracy: 98.63%')
        addLog('Model saved to models/minirocket_master_20241018.pth')
        clearInterval(interval)
      } else {
        setProgress(currentProgress)
        if (Math.random() > 0.7) {
          addLog(`Epoch ${Math.floor(currentProgress / 10)}: loss ${(Math.random() * 0.5).toFixed(4)}, acc ${(0.85 + Math.random() * 0.1).toFixed(3)}`)
        }
      }
    }, 300)
  }

  const resetTraining = () => {
    setIsTraining(false)
    setStatus('idle')
    setProgress(0)
    setLogs([])
  }

  return (
    <div className="main-content">
      <div className="page-header">
        <h1 className="page-title">Training Console</h1>
        <p className="page-subtitle">Configure and train motor imagery classification models</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-1.5rem">
        {/* Configuration Panel */}
        <div className="lg:col-span-1">
          <div className="card">
            <div className="card-header">
              <h3 className="card-title">Model Configuration</h3>
            </div>
            <div className="card-body space-y-3">
              <div className="form-group">
                <label className="form-label">Model Architecture</label>
                <select className="form-select" onChange={(e) => setConfig({ ...config, architecture: e.target.value as any })}>
                  <option value="minirocket">MiniRocket</option>
                  <option value="eegnet">EEGNet</option>
                  <option value="cnn_lstm">CNN-LSTM</option>
                  <option value="cnn_transformer">CNN+Transformer</option>
                </select>
              </div>
              <div className="form-group">
                <label className="form-label">Number of Kernels</label>
                <input type="number" className="form-input" value={config.numKernels} onChange={(e) => setConfig({ ...config, numKernels: parseInt(e.target.value) || 10000 })} />
              </div>
              <div className="form-group">
                <label className="form-label">Hidden Units</label>
                <input type="number" className="form-input" value={config.hidden} onChange={(e) => setConfig({ ...config, hidden: parseInt(e.target.value) || 512 })} />
              </div>
              <div className="form-group">
                <label className="form-label">Dropout Rate</label>
                <input type="number" className="form-input" step="0.05" value={config.dropout} onChange={(e) => setConfig({ ...config, dropout: parseFloat(e.target.value) || 0.15 })} />
              </div>
              <div className="form-group">
                <label className="form-label">Training Epochs</label>
                <input type="number" className="form-input" value={config.epochs} onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) || 30 })} />
              </div>
              <div className="form-group">
                <label className="form-label">Batch Size</label>
                <input type="number" className="form-input" value={config.batchSize} onChange={(e) => setConfig({ ...config, batchSize: parseInt(e.target.value) || 256 })} />
              </div>
              <div className="form-group">
                <label className="form-label">Learning Rate</label>
                <input type="number" className="form-input" step="0.0001" value={config.learningRate} onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) || 5e-4 })} />
              </div>
              <div className="form-group">
                <label className="form-label">Subject Range</label>
                <input type="text" className="form-input" value={config.subjects} onChange={(e) => setConfig({ ...config, subjects: e.target.value })} />
              </div>
            </div>
          </div>

          <div className="card mt-4">
            <div className="card-header">
              <h3 className="card-title">Dataset Preview</h3>
            </div>
            <div className="card-body">
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">Selected Subjects:</span>
                  <span className="font-mono text-slate-900">{config.subjects}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">Channels:</span>
                  <span className="font-mono text-slate-900">20</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">Sampling Rate:</span>
                  <span className="font-mono text-slate-900">160 Hz</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">Window Length:</span>
                  <span className="font-mono text-slate-900">4.0s (656 samples)</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600">Classes:</span>
                  <span className="font-mono text-slate-900">4</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Training Progress */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="card-header">
              <div className="flex items-center justify-between">
                <h3 className="card-title">Training Monitor</h3>
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${status === 'training' ? 'bg-emerald-500 animate-pulse' : 'bg-slate-300'}`}></span>
                  <span className="text-xs font-medium text-slate-600 capitalize">{status}</span>
                </div>
              </div>
            </div>
            <div className="card-body">
              <div className="mb-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-medium text-slate-700">Training Progress</span>
                  <span className="text-sm font-medium text-slate-700">{Math.round(progress)}%</span>
                </div>
                <div className="w-full bg-slate-200 rounded-full h-4">
                  <div className="bg-sky-600 h-full rounded-full transition-all duration-300" style={{ width: `${Math.min(progress, 100)}%` }}></div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 mb-4">
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <div className="text-xs text-slate-500 uppercase tracking-wider">Epochs Completed</div>
                  <div className="text-lg font-bold text-slate-900 font-mono">{Math.round((progress / 100) * config.epochs)}</div>
                </div>
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-200">
                  <div className="text-xs text-slate-500 uppercase tracking-wider">Estimated Time</div>
                  <div className="text-lg font-bold text-slate-900 font-mono">{Math.round((100 - progress) / 5)}s remaining</div>
                </div>
              </div>

              <div className="bg-slate-900 rounded-lg p-4 font-mono text-sm h-64 overflow-y-auto">
                {logs.map((log, idx) => (
                  <div key={idx} className="text-xs mb-1 text-emerald-400">{log}</div>
                ))}
              </div>

              <div className="mt-4 pt-4 border-t border-slate-200 flex justify-between gap-4">
                {isTraining ? (
                  <button className="btn btn-danger btn-sm" onClick={resetTraining}>Stop Training</button>
                ) : (
                  <>
                    <button className="btn btn-primary btn-sm" onClick={startTraining}>Start Training</button>
                    <button className="btn btn-outline btn-sm">Save Configuration</button>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}