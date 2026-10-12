import { useState, useEffect } from 'react'
import { getDatasets, submitTraining, getJobStatus, runPreview } from '../services/api'

export default function Training() {
  const [datasets, setDatasets] = useState([])
  const [selectedDataset, setSelectedDataset] = useState('')
  const [engine, setEngine] = useState('minirocket')
  const [epochs, setEpochs] = useState(10)
  const [batchSize, setBatchSize] = useState(256)
  const [learningRate, setLearningRate] = useState(0.001)
  const [job, setJob] = useState(null)
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)
  const [previewLoading, setPreviewLoading] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await getDatasets()
        setDatasets(res.data)
        if (res.data.length > 0) {
          setSelectedDataset(res.data[0].name)
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
      const res = await submitTraining({
        dataset: selectedDataset,
        engine,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate
      })
      setJob(res.data)
      setStatus(res.data)
    } catch (error) {
      console.error('Error submitting training:', error)
      alert('Error submitting training job')
    } finally {
      setLoading(false)
    }
  }

  const handleRunPreview = async () => {
    if (!job) return
    setPreviewLoading(true)
    try {
      const res = await runPreview(job.job_id, {
        dataset: selectedDataset,
        engine,
        epochs,
        batch_size: batchSize,
        learning_rate: learningRate
      })
      setStatus(res.data)
      setJob(res.data)
    } catch (error) {
      console.error('Error running preview:', error)
      alert('Error running preview')
    } finally {
      setPreviewLoading(false)
    }
  }

  const checkStatus = async () => {
    if (!job) return
    try {
      const res = await getJobStatus(job.job_id)
      setStatus(res.data)
    } catch (error) {
      console.error('Error getting status:', error)
    }
  }

  return (
    <div className="page-container">
      <h1>Training</h1>
      <p className="subtitle">Submit and manage training jobs</p>

      <div className="card">
        <h2>Submit Training Job</h2>
        <form onSubmit={handleSubmit} className="form-grid">
          <div className="form-group">
            <label>Dataset</label>
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
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
              <option value="eegnet">EEGNet</option>
              <option value="cnn_lstm">CNN-LSTM</option>
              <option value="convnets">ConvNets</option>
            </select>
          </div>

          <div className="form-group">
            <label>Epochs</label>
            <input
              type="number"
              value={epochs}
              onChange={(e) => setEpochs(parseInt(e.target.value))}
              min="1"
            />
          </div>

          <div className="form-group">
            <label>Batch Size</label>
            <input
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value))}
              min="1"
            />
          </div>

          <div className="form-group">
            <label>Learning Rate</label>
            <input
              type="number"
              step="0.0001"
              value={learningRate}
              onChange={(e) => setLearningRate(parseFloat(e.target.value))}
            />
          </div>

          <button type="submit" className="btn-primary" disabled={loading}>
            {loading ? 'Submitting...' : 'Submit Job'}
          </button>
        </form>
      </div>

      {job && (
        <div className="card">
          <h2>Job Status</h2>
          <div className="status-details">
            <p><strong>Job ID:</strong> {job.job_id}</p>
            <p><strong>Status:</strong> <span className={`status-${status?.status}`}>{status?.status}</span></p>
            <p><strong>Engine:</strong> {status?.engine}</p>
            <p><strong>Dataset:</strong> {status?.dataset}</p>
            <p><strong>Progress:</strong> {(status?.progress * 100).toFixed(2)}%</p>
            <p><strong>Message:</strong> {status?.message}</p>
          </div>
          <div className="button-group">
            <button onClick={checkStatus} className="btn-secondary">Refresh Status</button>
            <button
              onClick={handleRunPreview}
              className="btn-primary"
              disabled={previewLoading || status?.status === 'completed'}
            >
              {previewLoading ? 'Running...' : 'Run Preview (Synthetic)'}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
