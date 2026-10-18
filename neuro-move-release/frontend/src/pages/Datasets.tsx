import React, { useState } from 'react'

interface DatasetRow {
  id: string
  name: string
  type: string
  subjects: number
  channels: number
  samplingRate: number
  classes: string
  trials: number
  status: string
}

const datasets: DatasetRow[] = [
  { id: '1', name: 'PhysioNet EEGMMIDB', type: 'Motor Imagery', subjects: 109, channels: 64, samplingRate: 160, classes: 'Left Fist, Right Fist, Both Fists, Both Feet', trials: '18,000+', status: 'Ready' },
  { id: '2', name: 'BCI Competition IV 2a', type: 'Motor Imagery', subjects: 9, channels: 22, samplingRate: 250, classes: 'Left Hand, Right Hand, Both Feet, Tongue', trials: '288', status: 'Ready' },
  { id: '3', name: 'High-Gamma Dataset', type: 'Motor Imagery', subjects: 20, channels: 128, samplingRate: 5000, classes: 'Left Hand, Right Hand, Both Feet, Rest', trials: '992', status: 'Ready' },
  { id: '4', name: 'Kaya Finger Movements', type: 'Fine Motor Imagery', subjects: 13, channels: 22, samplingRate: 200, classes: 'Thumb, Index, Middle, Ring, Pinkie', trials: '15,000', status: 'Processing' },
  { id: '5', name: 'WAY-EEG-GAL', type: 'Grasp & Lift', subjects: 12, channels: 32, samplingRate: 512, classes: 'HandStart, FirstDigitTouch, LiftOff, Hold, Replace', trials: '3,936', status: 'Ready' },
  { id: '6', name: 'NEMAR Finger MI', type: 'Fine Motor Imagery', subjects: 20, channels: 65, samplingRate: 512, classes: 'Thumb, Index, Middle, Ring, Pinkie', trials: '15,000', status: 'Ready' },
]

export function Datasets() {
  const [showUpload, setShowUpload] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)

  return (
    <div className="main-content">
      <div className="page-header">
        <h1 className="page-title">Datasets</h1>
        <p className="page-subtitle">Manage EEG datasets for training and inference</p>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-sky-500 flex items-center justify-center text-white">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M12 4v7m0 0l3-3m-3 3L9 5M12 13v7m0 0l3-3m-3 3L9 17M5 12a7 7 0 1114 0 7 7 0 01-14 0z"/></svg>
              </div>
              <h3 className="card-title">Dataset Library</h3>
            </div>
            <div className="flex gap-2">
              <button className="btn btn-outline btn-sm">
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 3l2.3 2.3a2 2 0 010 2.8l-2.3 2.3M9 3l2.3 2.3a2 2 0 010 2.8l-2.3 2.3M21 12l2.3-2.3a2 2 0 000-2.8l-2.3-2.3M3 12l2.3-2.3a2 2 0 000-2.8l-2.3-2.3M21 12l2.3-2.3a2 2 0 000-2.8l-2.3-2.3M9 21l2.3-2.3a2 2 0 000-2.8l-2.3-2.3M3 21l2.3-2.3a2 2 0 000-2.8l-2.3-2.3M21 12l2.3-2.3a2 2 0 000-2.8l-2.3-2.3M9 12l2.3-2.3a2 2 0 000-2.8l-2.3-2.3M3 3l2.3-2.3a2 2 0 000-2.8l-2.3-2.3"/></svg>
                Upload Dataset
              </button>
              <button className="btn btn-secondary btn-sm">
                <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M12 5v14m0 0l6-6m-6 6l-6-6"/></svg>
                Refresh
              </button>
            </div>
          </div>
        </div>
        <div className="card-body">
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Dataset</th>
                  <th>Type</th>
                  <th className="text-right">Subjects</th>
                  <th className="text-right">Channels</th>
                  <th className="text-right">Sampling Rate</th>
                  <th>Classes</th>
                  <th className="text-right">Trials</th>
                  <th className="text-right">Status</th>
                  <th className="text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((ds) => (
                  <tr key={ds.id} className="hover:bg-slate-50">
                    <td className="font-medium text-slate-700">{ds.name}</td>
                    <td className="text-slate-600">{ds.type}</td>
                    <td className="text-right text-slate-600">{ds.subjects}</td>
                    <td className="text-right text-slate-600">{ds.channels}</td>
                    <td className="text-right text-slate-600">{ds.samplingRate} Hz</td>
                    <td className="text-slate-600">{ds.classes}</td>
                    <td className="text-right text-slate-600">{ds.trials}</td>
                    <td className="text-right">
                      <span className={`badge ${ds.status === 'Ready' ? 'badge-success' : 'badge-warning'}`}>{ds.status}</span>
                    </td>
                    <td className="text-right">
                      <button className="btn btn-outline btn-sm text-slate-600">
                        View
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {showUpload && (
        <div className="card mt-4">
          <div className="card-header">
            <h3 className="card-title">Upload New Dataset</h3>
          </div>
          <div className="card-body">
            <div className="file-upload" onClick={() => document.getElementById('file-input')?.click()}>
              <input type="file" id="file-input" accept=".edf,.gdf,.mat,.csv" onChange={(e) => setSelectedFile(e.target.files?.[0] || null)} />
              <div className="file-upload-icon">
                <svg className="w-5 h-5 text-sky-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
              </div>
              <p>Drop EDF, GDF, MAT, or CSV files here</p>
              <p className="file-hint">Supports .edf, .gdf, .mat, .csv formats</p>
            </div>
            {selectedFile && (
              <div className="mt-3 p-3 rounded-lg bg-sky-50 border border-sky-200">
                <p className="text-sm text-slate-700">Selected: {selectedFile.name} ({formatFileSize(selectedFile.size)})</p>
                <button className="mt-2 btn btn-primary btn-sm">Upload Dataset</button>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="mt-4">
        <button className="btn btn-primary" onClick={() => setShowUpload(!showUpload)}>
          {showUpload ? 'Close Upload' : '+ Upload New Dataset'}
        </button>
      </div>
    </div>
  )
}

function formatFileSize(bytes: number) {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}