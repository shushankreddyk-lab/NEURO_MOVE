import EEG3DVisualization from '../components/EEG3DVisualization'

export default function Visualization() {
  return (
    <div className="page-container">
      <h1>3D Brain Visualization</h1>
      <p className="subtitle">EEG Channel Topology - 10-20 International System</p>

      <div className="card">
        <h2>EEG Electrode Positions</h2>
        <p>Interactive 3D visualization of EEG electrode placements. Motor-relevant channels (C3, C4, Cz, etc.) are highlighted in cyan.</p>
        <EEG3DVisualization />
        <div className="viz-legend">
          <div className="legend-item">
            <span className="legend-color motor"></span>
            <span>Motor-related channels</span>
          </div>
          <div className="legend-item">
            <span className="legend-color frontal"></span>
            <span>Frontal channels</span>
          </div>
          <div className="legend-item">
            <span className="legend-color other"></span>
            <span>Other channels</span>
          </div>
        </div>
      </div>

      <div className="card">
        <h2>About the Visualization</h2>
        <p>
          This 3D visualization displays the standard 10-20 electrode placement system used in EEG recordings.
          Motor imagery tasks primarily activate the sensorimotor cortex, which is covered by central (C) and
          centroparietal (CP) electrodes highlighted in cyan.
        </p>
      </div>
    </div>
  )
}
