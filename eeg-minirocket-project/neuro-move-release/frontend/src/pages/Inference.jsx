import { useState, useEffect } from 'react'
import { getDatasets, infer } from '../services/api'

export default function Inference() {
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState('')
  const [engine, setEngine] = useState('minirocket')
  const [sampleCount, setSampleCount] = useState(32)
  const [channels, setChannels] = useState(22)
  const [samples, setSamples] = useState(656)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await getDatasets()
        setDatasets(res.data)
        if (res.data.length > 0) {
          setSelectedDataset(res.data[0].name)
          setChannels(res.data[0].default_channels)
          setSamples(res.data[0].default_samples)
        }
      } catch (error) {
        console.error('Error fetching datasets:', error)
      }
    }
    fetchData()
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await infer({
        engine,
        dataset: selectedDataset,
        sample_count: sampleCount,
        channels,
        samples
      })
      setResult(res.data)
    } catch (error) {
      console.error('Error running inference:', error)
      alert('Error running inference')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page-container">
      <h1>Inference</h1>
      <p className="subtitle">Run inference on synthetic EEG data</p>

      <div className="card">
        <h2>Inference Parameters</h2>
        <form onSubmit={handleSubmit} className="form-grid">
          <div className="form-group">
            <label>Dataset</label>
            <select
              value={selectedDataset}
              onChange={(e) => {
                setSelectedDataset(e.target.value)
                const ds = datasets.find(d => d.name === e.target.value)
                if (ds) {
                  setChannels(ds.default_channels)
                  setSamples(ds.default_samples)
                }
              }}
            >
              {datasets.map((ds, idx) => (
                <option key={idx} value={ds.name}>{ds.name}</option>
              ))}
            </select>
          </div>

          <div className="form-group">
            <label>Engine</label>
            <select value={engine} onChange={(e) => setEngine(e.target.value)}>
              <option value="minirocket">MiniRocket</option>
              <option value="conformer">Conformer</option>
            </select>
          </div>

          <div className="form-group">
            <label>Number of Samples</label>
            <input
              type="number"
              value={sampleCount}
              onChange={(e) => setSampleCount(parseInt(e.target.value))}
              min="1"
            />
          </div>

          <div className="form-group">
            <label>Channels</label>
            <input
              type="number"
              value={channels}
              onChange={(e) => setChannels(parseInt(e.target.value))}
              min="1"
            />
          </div>

          <div className="form-group">
            <label>Samples per Epoch</label>
            <input
              type="number"
              value={samples}
              onChange={(e) => setSamples(parseInt(e.target.value))}
              min="1"
            />
          </div>

          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? 'Running...' : 'Run Inference'}
          </button>
        </form>
      </div>

      {result && (
        <div className="card">
          <h2>Inference Results</h2>
          <div className="status-details">
            <p><strong>Status:</strong> <span className={result.ok ? 'status-ok' : 'status-failed'}>{result.ok ? 'Success' : 'Failed'}</span></p>
            <p><strong>Engine:</strong> {result.engine}</p>
            <p><strong>Dataset:</strong> {result.dataset}</p>
            <p><strong>Number of Samples:</strong> {result.num_samples}</p>
            <p><strong>Classes:</strong> {result.classes?.join(', ')}</p>
            <p><strong>Message:</strong> {result.message}</p>
          </div>
          {result.probabilities && (
            <div className="probabilities">
              <h3>Sample Probabilities</h3>
              <div className="prob-grid">
                {result.probabilities.slice(0, 4).map((prob, idx) => (
                  <div key={idx} className="prob-sample">
                    <strong>Sample {idx + 1}:</strong>
                    <div className="prob-bars">
                      {prob.map((p, cidx) => (
                        <div key={cidx} className="prob-bar">
                          <span>Class {cidx}:</span>
                          <div className="bar-bg">
                            <div
                              className="bar-fill"
                              style={{ width: `${p * 100}%` }}
                            ></div>
                          </div>
                          <span>{(p * 100).toFixed(2)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
