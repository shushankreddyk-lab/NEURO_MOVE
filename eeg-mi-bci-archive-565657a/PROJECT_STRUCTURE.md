# NeuroMove Enterprise MI-BCI - Project Structure

## Repository Layout

```
eeg-mi-bci-archive-565657a/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI/CD
├── api/                              # Production FastAPI Server
│   ├── main.py                       # FastAPI app with REST + WebSocket
│   ├── requirements.txt              # API dependencies
│   └── logging_utils.py              # Structured logging middleware
├── config/                           # Configuration Files
│   ├── logging.yaml                  # Structured logging config
│   └── postgres-init.sql             # MLflow DB initialization
├── dashboard/                        # Streamlit Enterprise Dashboard
│   └── app.py                        # 11-tab dashboard with real-time inference
├── docker-compose.yml                # Production Docker Compose
├── Dockerfile                        # Multi-stage hardened Dockerfile
├── mlflow/                           # MLflow Integration
│   ├── mlflow_tracker.py             # Experiment tracking & model registry
│   └── requirements.txt              # MLflow dependencies
├── monitoring/                       # Observability Stack
│   ├── prometheus.yml                # Prometheus scrape config
│   ├── alertmanager.yml              # Alert routing config
│   ├── grafana-dashboard.json        # Pre-built Grafana dashboard
│   ├── grafana-provisioning/         # Auto-provision datasources/dashboards
│   │   ├── datasources/prometheus.yaml
│   │   └── dashboards/neuromove.yaml
│   └── rules/
│       └── neuromove.yml             # Prometheus alerting rules
├── scripts/                          # Utility Scripts
│   ├── generate_synthetic.py         # Synthetic EEG data generator
│   ├── verify_datasets.py            # Dataset verification & download info
│   ├── overtrain_targets.py          # Model overtraining utility
│   └── training_process_snippet.py   # Dashboard snippet
├── src/                              # Core Engine Library
│   ├── __init__.py                   # Unified exports
│   ├── config_loader.py              # YAML config + env substitution
│   ├── minirocket_engine.py          # MiniRocket + GPU MLP Head (FIXED)
│   ├── eegnet_engine.py              # EEGNet (Lawhern et al.)
│   ├── cnn_lstm_engine.py            # CNN-LSTM hybrid
│   ├── convnets_engine.py            # Shallow/Deep ConvNet
│   ├── advanced_eeg_engine.py        # Transformer/Conformer
│   ├── transformer_engine.py         # Pure Transformer
│   ├── preprocessing.py              # Bandpass, CAR, z-score
│   ├── binary_parser.py              # EDF/GDF/MAT loaders
│   ├── dataset_loader_all.py         # Unified dataset interface
│   ├── dataset_2a_loader.py          # BCI 2a specific loader
│   ├── model_registry.py             # Thread-safe model registry
│   ├── evaluation.py                 # LOSO CV + metrics
│   ├── benchmark.py                  # Multi-architecture benchmark runner
│   ├── train_master.py               # Master training script (FIXED)
│   ├── visualizer.py                 # Plotting utilities
│   ├── engines_safe_print.py         # Safe print utility
│   ├── realtime_inference_sim.py     # Real-time inference simulator
│   ├── exploratory_analysis.py       # Exploratory data analysis
│   └── sitecustomize.py              # Site customization
├── tests/                            # Test Suite
│   └── test_core.py                  # 17 tests (15 passed, 2 skipped)
├── models/                           # Trained Model Weights (.pth)
├── artifacts/                        # Training caches, logs
├── figures/                          # Architecture diagrams
├── reports/                          # Generated reports
├── data/                             # Dataset storage (gitignored)
├── mlruns/                           # MLflow tracking (gitignored)
├── logs/                             # Application logs (gitignored)
├── .env.example                      # Environment template
├── .gitignore                        # Comprehensive ignore rules
├── .pre-commit-config.yaml           # Ruff, Black, isort hooks
├── pyproject.toml                    # Build + tool config (ruff, pytest, coverage)
├── requirements.txt                  # Core dependencies
├── README.md                         # Project documentation
├── DEPLOYMENT.md                     # Deployment guide
├── PROJECT_STRUCTURE.md              # This file
├── config.yaml                       # Centralized enterprise config
└── config_loader.py                  # (Legacy - use src/config_loader.py)
```

---

## Key Directories Explained

### `/src` - Core Engine Library (Canonical)
All 15 neural architectures + utilities. **Single source of truth** for model implementations.

| File | Purpose |
|------|---------|
| `minirocket_engine.py` | MiniRocket + GPU MLP Head (primary engine) |
| `eegnet_engine.py` | EEGNet compact CNN |
| `cnn_lstm_engine.py` | CNN-LSTM hybrid for temporal modeling |
| `convnets_engine.py` | Shallow & Deep ConvNet (Schirrmeister) |
| `advanced_eeg_engine.py` | Transformer/Conformer |
| `transformer_engine.py` | Pure Transformer |
| `preprocessing.py` | Bandpass, CAR, z-score, epoching |
| `binary_parser.py` | EDF/GDF/MAT/CSV loaders |
| `dataset_loader_all.py` | Unified multi-dataset interface |
| `dataset_2a_loader.py` | BCI Competition IV 2a loader |
| `model_registry.py` | Thread-safe model registry with atomic writes |
| `evaluation.py` | LOSO CV, per-subject metrics |
| `benchmark.py` | Automated multi-architecture benchmarking |
| `train_master.py` | Master training orchestration |
| `config_loader.py` | YAML config with `${ENV_VAR}` substitution |
| `visualizer.py` | Topomaps, confusion matrices, training curves |

### `/api` - Production REST + WebSocket Server
FastAPI-based inference server with:
- REST endpoints for batch inference
- WebSocket for real-time streaming
- Model registry integration
- Prometheus metrics (`/metrics/prometheus`)
- Structured JSON logging
- Bearer token authentication

### `/dashboard` - Streamlit Enterprise UI
11-tab dashboard:
1. **Overview** - Project abstract, challenges, methodology
2. **Model Architectures** - Benchmark matrix, Pareto frontier, radar charts
3. **Live Training Console** - Real-time training monitoring
4. **Live Training** - Hyperparameter tuning
5. **Training Process** - Architecture deep-dive with Mermaid diagrams
6. **Preprocessing** - Pipeline visualization
7. **Signal Analysis** - Multi-channel EEG, PSD, ERD/ERS
8. **Live Inference** - File upload + real-time prediction
9. **Accuracy Analysis** - Per-subject, confusion matrices
10. **Global Analytics** - Cross-dataset comparison
11. **Technical Details** - Architecture specs

### `/mlflow` - Experiment Tracking
MLflow wrapper with:
- `NeuroMoveMLflowTracker` - Experiment/run management
- `TrainingRun` context manager for automatic run lifecycle
- Model registry with champion/challenger aliases
- Dataset metadata logging
- Training curve artifact logging

### `/monitoring` - Full Observability
- **Prometheus**: 10s scrape interval, custom rules
- **Grafana**: Pre-built dashboard (latency, throughput, model status, WS connections)
- **Alertmanager**: Email/Slack/PagerDuty routing with severity tiers
- **Node Exporter + cAdvisor**: Host + container metrics

### `/scripts` - Operational Utilities
- `generate_synthetic.py` - CI-ready synthetic EEG (PhysioNet, BCI2a, High-Gamma, etc.)
- `verify_datasets.py` - Checks 6 datasets, provides download instructions
- `overtrain_targets.py` - Overtraining utility for target datasets
- `training_process_snippet.py` - Dashboard code fragment

---

## Configuration

### `config.yaml` - Single Source of Truth
All runtime parameters centralized:
```yaml
minirocket:
  num_kernels: 10000
  head_epochs: 30
  head_lr: 5e-4

preprocessing:
  sfreq: 160
  low_freq: 4.0
  high_freq: 38.0
  apply_car: true

dashboard:
  page_title: "NeuroMove Enterprise"
  theme: "dark"
```

Override via environment:
```bash
export NEUROMOVE_ROOT=/custom/path
export NEUROMOVE_PIP_PACKAGES=/opt/cuda/torch
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test class
pytest tests/test_core.py::TestConfigLoader -v
```

Current status: **15 passed, 2 skipped** (torch CUDA DLL is Windows env issue, not code)

---

## Docker Deployment

### Development
```bash
docker-compose up -d
```

### Production (with overrides)
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Build Only
```bash
docker build -t neuromove:latest .
```

---

## CI/CD Pipeline

`.github/workflows/ci.yml` stages:
1. **Lint** - `ruff check .` + `ruff format --check .`
2. **Test** - `pytest` with coverage
3. **Smoke** - Import verification + config load test
4. **Docker** - Multi-stage build verification

---

## Enterprise Standards Compliance

| Standard | Implementation |
|----------|----------------|
| **Code Quality** | Ruff (lint + format), Black, isort, pre-commit |
| **Type Safety** | Pydantic models, type hints in public APIs |
| **Security** | Non-root containers, Bearer auth, secret env vars |
| **Observability** | Prometheus metrics, Grafana dashboards, structured JSON logs |
| **Reliability** | Health/readiness probes, graceful shutdown, circuit breakers |
| **Reproducibility** | MLflow tracking, config-driven, pinned dependencies |
| **Scalability** | Stateless API, horizontal scaling ready, Redis caching |
| **Compliance** | Audit logs, model versioning, immutable registry |

---

## File Ownership & Permissions

All Docker volumes owned by `neuromove:1000` (non-root user):
```
neuromove_models    → /app/models
neuromove_artifacts → /app/artifacts
neuromove_logs      → /app/logs
neuromove_mlruns    → /app/mlruns
neuromove_postgres  → /var/lib/postgresql/data
neuromove_redis     → /data
neuromove_prometheus → /prometheus
neuromove_grafana   → /var/lib/grafana
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-10-24 | Initial enterprise release with full stack |

---

## Support

- **Architecture Questions**: See `DEPLOYMENT.md`
- **API Reference**: `http://localhost:8000/docs`
- **Dashboard**: `http://localhost:8501`
- **Monitoring**: `http://localhost:3000` (Grafana)