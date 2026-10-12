import { useState, useEffect } from 'react'
import { getDatasets } from '../services/api'

export default function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await getDatasets()
        setDatasets(res.data)
      } catch (error) {
        console.error('Error fetching datasets:', error)
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
        <p>Loading datasets...</p>
      </div>
    )
  }

  return (
    <div className="page-container">
      <h1>Datasets</h1>
      <p className="subtitle">Available EEG Motor Imagery Datasets</p>

      <div className="datasets-list">
        {datasets.map((ds, idx) => (
          <div key={idx} className="card dataset-detail">
            <h2>{ds.name}</h2>
            <p className="description">{ds.description}</p>
            <div className="dataset-stats">
              <div>
                <strong>Channels:</strong> {ds.default_channels}
              </div>
              <div>
                <strong>Samples:</strong> {ds.default_samples}
              </div>
              <div>
                <strong>Classes:</strong> {ds.num_classes}
              </div>
            </div>
            <p className={`status ${ds.directory_exists ? 'ok' : 'missing'}`}>
              {ds.directory_exists ? '✓ Data directory found' : '✗ Data directory not found'}
            </p>
            <small>Path hint: {ds.dir_hint}</small>
          </div>
        ))}
      </div>
    </div>
  )
}
