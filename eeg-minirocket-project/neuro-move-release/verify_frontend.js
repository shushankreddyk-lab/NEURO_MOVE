const fs = require('fs')
const path = require('path')

const ROOT = 'D:/eeg-minirocket-project/neuro-move-release/frontend'
const errors = []

function checkExists(file, label) {
  const full = path.join(ROOT, file)
  if (!fs.existsSync(full)) errors.push(`missing: ${label} (${file})`)
}

function checkText(file, pattern) {
  const full = path.join(ROOT, file)
  try {
    const text = fs.readFileSync(full, 'utf8')
    if (!pattern.test(text)) errors.push(`missing text in ${file}: ${pattern}`)
  } catch (e) {
    errors.push(`read error ${file}: ${e.message}`)
  }
}

// Required files
checkExists('package.json', 'package.json')
checkExists('index.html', 'index.html')
checkExists('vite.config.js', 'vite.config.js')
checkExists('src/main.jsx', 'src/main.jsx')
checkExists('src/App.jsx', 'src/App.jsx')
checkExists('src/App.css', 'src/App.css')
checkExists('src/index.css', 'src/index.css')
checkExists('src/services/api.js', 'src/services/api.js')
checkExists('src/pages/Dashboard.jsx', 'src/pages/Dashboard.jsx')
checkExists('src/pages/Datasets.jsx', 'src/pages/Datasets.jsx')
checkExists('src/pages/Training.jsx', 'src/pages/Training.jsx')
checkExists('src/pages/Inference.jsx', 'src/pages/Inference.jsx')
checkExists('src/pages/Visualization.jsx', 'src/pages/Visualization.jsx')
checkExists('src/components/EEG3DVisualization.jsx', 'src/components/EEG3DVisualization.jsx')

// React Router wired in App
checkText('src/App.jsx', /import.*Routes.*from 'react-router-dom'/)
checkText('src/App.jsx', /<Routes>/)
checkText('src/App.jsx', /<Route path="\/datasets" element={<Datasets \/>} \/>/)
checkText('src/App.jsx', /<Route path="\/training" element={<Training \/>} \/>/)
checkText('src/App.jsx', /<Route path="\/inference" element={<Inference \/>} \/>/)
checkText('src/App.jsx', /<Route path="\/visualization" element={<Visualization \/>} \/>/)

// Routing setup in main
checkText('src/main.jsx', /import.*BrowserRouter.*from 'react-router-dom'/)

// API endpoints planned in pages
checkText('src/pages/Dashboard.jsx', /getHealth\(\)/)
checkText('src/pages/Dashboard.jsx', /getDatasets\(\)/)
checkText('src/pages/Datasets.jsx', /getDatasets\(\)/)
checkText('src/pages/Training.jsx', /submitTraining\(/)
checkText('src/pages/Training.jsx', /getJobStatus\(/)
checkText('src/pages/Training.jsx', /runPreview\(/)
checkText('src/pages/Inference.jsx', /infer\(/)

// 3D component structure
checkText('src/components/EEG3DVisualization.jsx', /import.*Canvas.*from '@react-three\/fiber'/)
checkText('src/components/EEG3DVisualization.jsx', /import.*OrbitControls.*from '@react-three\/drei'/)
checkText('src/components/EEG3DVisualization.jsx', /<Canvas/)

// package.json dependencies present
try {
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'))
  const required = [
    '@react-three/drei',
    '@react-three/fiber',
    'axios',
    'react-router-dom',
    'recharts',
    'three',
  ]
  for (const dep of required) {
    if (pkg.dependencies[dep] == null) errors.push(`missing dependency: ${dep}`)
  }
} catch (e) {
  errors.push(`package.json parse: ${e.message}`)
}

// index.html has root mount and module script
checkText('index.html', /<div id="root"><\/div>/)
checkText('index.html', /<script type="module".*src="\/src\/main\.jsx"><\/script>/)

console.log('Frontend verification complete.')
if (errors.length) {
  console.error('Errors:')
  errors.forEach((e) => console.error(' - ' + e))
  process.exit(1)
} else {
  console.log('No errors.')
  process.exit(0)
}
