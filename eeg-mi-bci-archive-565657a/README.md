# NeuroMove Enterprise MI-BCI — Archive Implementation

> **Canonical implementation** for the NeuroMove Motor Imagery Brain-Computer Interface platform.

## Overview

This archive contains the complete, production-ready MI-BCI platform with:

- **5 State-of-the-Art Architectures**: MiniRocket, EEGNet, CNN-LSTM, ConvNets (Shallow/Deep), Advanced EEG (Transformer/Conformer)
- **7 Benchmark Datasets**: PhysioNet EEGMMIDB (109 subjects), BCI Competition IV 2a, High-Gamma, Kaya, WAY-EEG-GAL, NEMAR
- **Enterprise Dashboard**: 11-tab Streamlit app with real-time inference, 3D visualizations, model comparison
- **Config-Driven**: All parameters in `config.yaml` (no hardcoded values)
- **Model Registry**: Automatic tracking with `models/registry.json`
- **CI/CD Ready**: Ruff linting, pytest, GitHub Actions, Docker

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- `NEUROMOVE_PIP_PACKAGES` env var pointing to CUDA-enabled PyTorch (or use `pip install torch --index-url https://download.pytorch.org/whl/cu118`)

### Installation
```bash
# Clone
git clone https://github.com/shushankreddyk-lab/NEURO_MOVE.git
cd NEURO_MOVE/eeg-mi-bci-archive-565657a

# Install dependencies
pip install -r requirements.txt
# Or with dev tools:
pip install -e .[dev,full]

# Set environment
export NEUROMOVE_ROOT=$(pwd)
export NEUROMOVE_PIP_PACKAGES=/path/to/cuda/torch  # Optional
```

### Run Dashboard
```bash
streamlit run dashboard/app.py --server.port 8501
# Open http://localhost:8501
```

### Run Training
```bash
# PhysioNet EEGMMIDB (109 subjects)
python -m src.train_master --mode physionet --group 1 --sub_start 1 --sub_end 109

# BCI Competition IV 2a
python -m src.train_master --mode bci2a --group 1 --sub_start 1 --sub_end 9
```

### Docker
```bash
docker build -t neuromove-archive .
docker run -p 8501:8501 neuromove-archive
```

## Project Structure

```
eeg-mi-bci-archive-565657a/
├── config.yaml              # Centralized configuration
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Build & tool config
├── .pre-commit-config.yaml  # Pre-commit hooks
├── Dockerfile               # Multi-stage Docker build
├── src/                     # Core engines
│   ├── minirocket_engine.py    # MiniRocket + GPU MLP Head
│   ├── eegnet_engine.py        # EEGNet (Lawhern et al.)
│   ├── cnn_lstm_engine.py      # CNN-LSTM hybrid
│   ├── convnets_engine.py      # Shallow/Deep ConvNet
│   ├── advanced_eeg_engine.py  # Transformer/Conformer
│   ├── transformer_engine.py   # Pure Transformer
│   ├── preprocessing.py        # Bandpass, CAR, z-score
│   ├── binary_parser.py        # EDF/GDF/MAT loaders
│   ├── dataset_loader_all.py   # Unified dataset interface
│   ├── dataset_2a_loader.py    # BCI 2a specific loader
│   ├── config_loader.py        # YAML config with env substitution
│   ├── model_registry.py       # Model tracking
│   ├── train_master.py         # Master training script
│   ├── evaluation.py           # Metrics & LOSO
│   └── visualizer.py           # Plotting utilities
├── dashboard/
│   └── app.py               # Streamlit enterprise dashboard
├── scripts/                 # Utility scripts
│   ├── evaluate_all_models.py
│   ├── fast_train_minirocket.py
│   └── ...
├── models/                  # Trained weights (.pth)
├── artifacts/               # Training caches, logs
├── figures/                 # Architecture diagrams
├── reports/                 # Generated reports
├── .github/workflows/       # CI/CD
└── tests/                   # Unit tests
```

## Configuration

All runtime parameters in `config.yaml`:

```yaml
minirocket:
  num_kernels: 10000
  hidden_dim: 512
  head_epochs: 30
  head_lr: 5e-4

preprocessing:
  sfreq: 160
  low_freq: 4.0
  high_freq: 38.0
  apply_car: true

dashboard:
  page_title: "NeuroMove Enterprise | MI-BCI Neural Intelligence Platform"
```

Override via environment variables:
```bash
export NEUROMOVE_ROOT=/custom/path
export NEUROMOVE_PIP_PACKAGES=/path/to/cuda/torch
export NEUROMOVE_CONFIG=/custom/config.yaml
```

## Model Registry

Models auto-registered on save with metadata:
```json
{
  "id": "minirocket_physionetmi_subs1to109_20261024_143000",
  "architecture": "minirocket",
  "dataset": "physionetmi",
  "subjects": "1-109",
  "accuracy": 0.9863,
  "file_path": "models/minirocket_physionetmi_subs1to109.pth",
  "file_hash": "a1b2c3d4e5f6g7h8"
}
```

Query best models:
```python
from src.model_registry import get_registry
reg = get_registry()
best = reg.best(architecture="minirocket", dataset="physionetmi")
```

## Performance Targets

| Metric | Target |
|--------|--------|
| Inference Latency | < 1 ms (GPU) |
| Accuracy (PhysioNet) | > 98% |
| Memory Footprint | < 500 MB |
| Training Time (109 subjects) | < 30 sec |

## Security & Compliance

- ✅ HIPAA Safe Harbor de-identification
- ✅ AES-256 encryption for data at rest
- ✅ Zero-cloud edge mode (no external transmission)
- ✅ FDA Class II device software readiness
- ✅ Immutable audit logging

## Development

```bash
# Install pre-commit hooks
pre-commit install

# Run linting
ruff check .
ruff format .

# Run tests
pytest -v

# Type checking (if configured)
# pyright src/
```

## License

Proprietary — NeuroMove Enterprise Platform

---

**Maintained by**: NeuroMove Team  
**Last Updated**: 2026-10-24