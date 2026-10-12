# NeuroMove: Advanced Multi-Modal Motor Imagery BCI Platform

An end-to-end Motor Imagery Brain-Computer Interface (MI-BCI) research and clinical intelligence system integrating state-of-the-art Deep Learning (MiniRocket, EEGNet, Shallow ConvNet, CNN-LSTM, Conformer Transformer) with real-time 3D neuro-signal analytics.

---

## 📁 Repository Directory Structure

The project has been organized into modular, classified subdirectories:

```
eeg-minirocket-project/
├── eeg-mi-bci/             # Core BCI processing engines & active Streamlit 3D interactive dashboard
│   ├── dashboard/          # Streamlit UI (app.py) & Three.js 3D WebGL components
│   ├── src/                # Deep learning & signal processing engines (MiniRocket, EEGNet, Conformer, etc.)
│   ├── models/             # Production trained weights (.pth and .pkl)
│   ├── artifacts/          # Benchmarking metrics, evaluation results, and training caches
│   └── tests/              # Unit & integration tests for BCI pipelines
│
├── neuro-move-release/     # Standalone Production Web Platform (FastAPI Backend + React 18 / Three.js Frontend)
│   ├── backend/            # High-throughput FastAPI REST API & real-time WebSocket inference server
│   └── frontend/           # Modern 3D glassmorphic/liquid UI with Three.js EEG scalp topology
│
├── reports/                # Project publications, comprehensive reports, and thesis drafts
│   ├── FINAL_PROJECT_REPORT_FOR_MAM.pdf   # 20+ Page Comprehensive Project Report (PDF)
│   ├── FINAL_PROJECT_REPORT_FOR_MAM.html  # Complete Interactive Project Report (HTML)
│   ├── FINAL_20_PAGE_IEEE_REPORT.html     # Formal IEEE Transactions Format Report
│   ├── Elsevier_Base_Paper_Clone.html     # Comparative Base Paper Clone
│   ├── Thesis_Draft.docx                  # Full Thesis Manuscript Draft
│   └── project_report.md                  # Markdown project executive summary
│
├── figures/                # High-resolution architectural diagrams, flowcharts & dataset topologies
│   ├── three_tier_architecture_*.jpg      # End-to-end 3-tier architectural schematic
│   ├── minirocket_arch_*.jpg              # MiniRocket kernel transform diagram
│   ├── eegnet_arch_*.jpg                  # EEGNet depthwise separable convolution architecture
│   ├── transformer_arch_*.jpg             # Multi-head attention Conformer architecture
│   └── *_ds_*.jpg                         # Dataset protocol topographies (PhysioNet, BCI IV-2a, WAY-EEG, etc.)
│
├── scripts/                # Data downloaders, training orchestration, and test utilities
│   ├── auto_train_all.py                  # End-to-end batch training supervisor
│   ├── download_moabb_datasets.py         # MOABB multi-dataset ingestion pipeline
│   ├── download_nemar_on008446.py         # OpenNeuro NEMAR dataset downloader
│   ├── generate_report.py                 # IEEE report automated builder
│   └── smoke_test.py                      # System health verification script
│
├── dataset/                # Harmonized motor imagery EEG datasets (PhysioNet, BCI-2a, etc.)
├── models/                 # Global model weights repository
├── requirements.txt        # Python dependency manifest
└── .gitignore              # Git ignore configuration
```

---

## 🚀 Quick Start

### 1. Interactive 3D Streamlit Dashboard
```bash
cd eeg-mi-bci
streamlit run dashboard/app.py
```

### 2. Production React + FastAPI Platform
```bash
# Start FastAPI Backend
cd neuro-move-release/backend
python run.py

# Start React Frontend
cd neuro-move-release/frontend
npm install
npm run dev
```

---

## 🧠 Supported Neural Architectures
- **MiniRocket (GPU / RidgeClassifier)**: Ultra-fast random convolutional kernel feature transform.
- **EEGNet**: Compact depthwise and separable convolutional network for P300, ERD, and ERS decoding.
- **Shallow ConvNet**: Bandpower & spatial filter mimicry inspired by Filter Bank CSP.
- **CNN-LSTM**: Joint spatio-temporal representation learning for continuous motor execution.
- **Conformer / Transformer**: Multi-head self-attention network for long-range temporal dependencies.
