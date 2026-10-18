import { Routes, Route, NavLink } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Datasets from './pages/Datasets'
import Training from './pages/Training'
import Inference from './pages/Inference'
import Visualization from './pages/Visualization'
import './App.css'

function App() {
  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-brand">
          <h2>🧠 NeuroMove</h2>
        </div>
        <ul className="nav-links">
          <li>
            <NavLink to="/" end className={({ isActive }) => isActive ? 'active' : ''}>
              Dashboard
            </NavLink>
          </li>
          <li>
            <NavLink to="/datasets" className={({ isActive }) => isActive ? 'active' : ''}>
              Datasets
            </NavLink>
          </li>
          <li>
            <NavLink to="/training" className={({ isActive }) => isActive ? 'active' : ''}>
              Training
            </NavLink>
          </li>
          <li>
            <NavLink to="/inference" className={({ isActive }) => isActive ? 'active' : ''}>
              Inference
            </NavLink>
          </li>
          <li>
            <NavLink to="/visualization" className={({ isActive }) => isActive ? 'active' : ''}>
              3D Visualization
            </NavLink>
          </li>
        </ul>
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/training" element={<Training />} />
          <Route path="/inference" element={<Inference />} />
          <Route path="/visualization" element={<Visualization />} />
        </Routes>
      </main>
    </div>
  )
}

export default App
