import { useState, useEffect } from 'react'
import { getHealth, getDatasets } from '../services/api'

export default function Dashboard() {
  const [health, setHealth] = useState(null)
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [healthRes, datasetsRes] = await Promise.all([
          getHealth(),
          getDatasets()
        ])
        setHealth(healthRes.data)
        setDatasets(datasetsRes.data)
      } catch (error) {
        console.error('Error fetching data:', error)
      } finally {
        setLoading(false)
      }
    }
    fetchData()
  }, [])

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading system status...</p>
      </div>
    )
  }

  return (
    <div className="page-container">
      <h1>NeuroMove MI-BCI System</h1>
      <p className="subtitle">Advanced Brain-Computer Interface for Motor Imagery Decoding</p>

      <div className="dashboard-grid">
        <div className="card">
          <h2>System Health</h2>
          {health && (
            <div className="status-info">
              <p><strong>Status:</strong> <span className="status-ok">Online</span></p>
              <p><strong>Python:</strong> {health.python}</p>
              <p><strong>Engines Available:</strong> {Object.keys(health.engines || {}).length}</p>
              <p><strong>Virtual Env:</strong> {health.venv_reachable ? 'Connected' : 'Not Found'}</p>
            </div>
          )}
        </div>

        <div className="card">
          <h2>Available Engines</h2>
          {health?.engines && Object.entries(health.engines).map(([name, info]) => (
            <div key={name} className="engine-item">
              <strong>{name}</strong>: {info.available ? 'Available' : 'Unavailable'}
              {info.error && <div className="error-text">{info.error}</div>}
            </div>
          ))}
        </div>

        <div className="card full-width">
          <h2>Datasets</h2>
          <div className="dataset-grid">
            {datasets.map((ds, idx) => (
              <div key={idx} className="dataset-item">
                <h3>{ds.name}</h3>
                <p><strong>Channels:</strong> {ds.default_channels}</p>
                <p><strong>Samples:</strong> {ds.default_samples}</p>
                <p><strong>Classes:</strong> {ds.num_classes}</p>
                <p className={ds.directory_exists ? 'exists' : 'not-exists'}>
                  {ds.directory_exists ? '✓ Directory Found' : '✗ Directory Missing'}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
