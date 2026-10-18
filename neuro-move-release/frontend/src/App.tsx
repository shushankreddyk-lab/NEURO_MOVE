import React from 'react'
import { HashRouter as Router, Routes, Route } from 'react-router-dom'
import { Dashboard } from './pages/Dashboard'
import { Datasets } from './pages/Datasets'
import { Training } from './pages/Training'
import { Inference } from './pages/Inference'
import { Models } from './pages/Models'
import { Analytics } from './pages/Analytics'
import { Settings } from './pages/Settings'
import './index.css'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-slate-50 text-slate-900 font-sans antialiased selection:bg-sky-200 selection:text-slate-900">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/training" element={<Training />} />
          <Route path="/inference" element={<Inference />} />
          <Route path="/models" element={<Models />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App