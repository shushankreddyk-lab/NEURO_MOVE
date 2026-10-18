import React from 'react'

const models = [
  { id: '1', name: 'MiniRocket_PhysioNet_109subs', architecture: 'MiniRocket', classes: 'Left Fist, Right Fist, Both Fists, Both Feet', accuracy: '98.63%', samples: '18,000+', status: 'Active', date: '2026-10-15' },
  { id: '2', name: 'CNN-LSTM_BCI2a_9subs', architecture: 'CNN-LSTM', classes: 'Left Hand, Right Hand, Both Feet, Tongue', accuracy: '98.06%', samples: '288', status: 'Active', date: '2026-10-14' },
  { id: '3', name: 'Conformer_HighGamma_20subs', architecture: 'CNN+Transformer', classes: 'Left Hand, Right Hand, Both Feet, Rest', accuracy: '96.50%', samples: '992', status: 'Active', date: '2026-10-13' },
  { id: '4', name: 'EEGNet_Kaya_13subs', architecture: 'EEGNet', classes: 'Thumb, Index, Middle, Ring, Pinkie', accuracy: '95.20%', samples: '15,000', status: 'Active', date: '2026-10-12' },
]

export function Models() {
  return (
    <div className="main-content">
      <div className="page-header">
        <h1 className="page-title">Model Registry</h1>
        <p className="page-subtitle">Manage trained models and their versions</p>
      </div>

      <div className="card">
        <div className="card-header">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-sky-500 flex items-center justify-center text-white">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M12 2L2 7l10 5 10-5-10-5-10 5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
              </div>
              <h3 className="card-title">Trained Models</h3>
            </div>
            <button className="btn btn-primary btn-sm">
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M12 4v16m0-16l4 4m-4 4l-4-4"/></svg>
              New Model
            </button>
          </div>
          <div className="card-body">
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Model Name</th>
                    <th>Architecture</th>
                    <th className="text-right">Accuracy</th>
                    <th className="text-right">Samples</th>
                    <th className="text-right">Classes</th>
                    <th className="text-right">Created</th>
                    <th className="text-right">Status</th>
                    <th className="text-right">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((m) => (
                    <tr key={m.id} className="hover:bg-slate-50">
                      <td className="font-medium text-slate-700">{m.name}</td>
                      <td className="text-slate-600">{m.architecture}</td>
                      <td className="text-right font-medium text-slate-600">{m.accuracy}</td>
                      <td className="text-right text-slate-600">{m.samples}</td>
                      <td className="text-slate-600">{m.classes}</td>
                      <td className="text-right text-slate-600">{m.date}</td>
                      <td className="text-right">
                        <span className="badge badge-success">Active</span>
                      </td>
                      <td className="text-right">
                        <div className="flex gap-2">
                          <button className="btn btn-outline btn-sm text-slate-600" title="View Details">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M9 12l2 2 4-4"/></svg>
                          </button>
                          <button className="btn btn-outline btn-sm text-slate-600" title="Download">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
                          </button>
                          <button className="btn btn-outline btn-sm text-red-600" title="Delete">
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth="2"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m4 0v12m0 0l4-4m-4 4l-4-4M17 1l-4 4m4 4l-4-4"/></svg>
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}