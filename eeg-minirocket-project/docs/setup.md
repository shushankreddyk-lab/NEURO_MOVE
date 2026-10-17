# Setup Guide — NeuroMove MI-BCI Platform

## 1. Prerequisites

- Python 3.10+
- Node.js 20+ (for the React frontend)
- CUDA-capable GPU (optional, for GPU-accelerated MiniRocket)
- Windows: MSVC C++ Build Tools (only if compiling `pyedflib`; we use pure-Python `edfio` by default)

## 2. Clone the Repository

```bash
git clone https://github.com/shushankreddyk-lab/NEURO_MOVE.git
cd NEURO_MOVE
```

## 3. Python Environment

Create and activate a virtual environment:

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The root `requirements.txt` covers the Streamlit dashboard and all training engines. It includes `streamlit-option-menu` and `streamlit-lottie` which are required by `eeg-mi-bci/dashboard/app.py`.

## 4. Datasets and Models (not in git)

The `.gitignore` excludes `data/`, `datasets/`, `dataset/`, `models/`, `results/`, and `*.pth`/`*.pkl`/`*.csv` files. You must place your data manually:

```
eeg-minirocket-project/
├── data/                 # Place raw EEG datasets here (PhysioNet, BCI-2a, etc.)
├── models/               # Place trained weights (.pth, .pkl) here
├── results/              # Training/evaluation outputs go here
└── eeg-mi-bci/
    ├── artifacts/        # Benchmark metrics, caches
    └── models/           # Per-experiment model weights
```

## 5. Running the Streamlit Dashboard (primary UI)

```bash
cd eeg-mi-bci
streamlit run dashboard/app.py
```

The dashboard imports `streamlit-option-menu` and `streamlit-lottie` — ensure they are installed (step 3).

To force a specific CUDA torch build, set the environment variable before launching:

```bash
# Windows (PowerShell)
$env:NEUROMOVE_PIP_PACKAGES="D:\pip_packages"
streamlit run dashboard/app.py
```

## 6. Running the FastAPI Backend

```bash
cd neuro-move-release/backend
python run.py
```

Backend endpoints:
- `GET /health`
- `GET /datasets`
- `POST /preprocess`
- `POST /train`
- `GET /train/{job_id}`
- `POST /train/{job_id}/run-preview`
- `POST /infer`
- `POST /evaluate`

The backend resolves paths dynamically (no hardcoded `D:\`). Override the project root with `NEUROMOVE_ROOT` if needed.

## 7. Running the React Frontend

```bash
cd neuro-move-release/frontend
npm install
npm run dev
```

Frontend connects to `http://localhost:8000` (FastAPI backend). Ensure the backend is running first.

## 8. Running Tests

```bash
# Install test deps
pip install pytest

# Run all tests
pytest

# Run a specific file
pytest tests/test_pipeline.py
```

Tests live in `tests/`. The `test_label_mapping.py` test requires PhysioNet `.edf` files on disk — it will skip/fail if data is absent.

## 9. Linting and Formatting

```bash
pip install ruff

# Lint
ruff check eeg-mi-bci scripts neuro-move-release/backend

# Format check
ruff format --check eeg-mi-bci scripts neuro-move-release/backend

# Auto-fix and format
ruff check --fix eeg-mi-bci scripts neuro-move-release/backend
ruff format eeg-mi-bci scripts neuro-move-release/backend
```

Pre-commit hooks are configured in `.pre-commit-config.yaml`.

## 10. CI

GitHub Actions runs on every push/PR to `main`:
- Python 3.10 / 3.11 / 3.12: `ruff check`, `ruff format --check`, `pytest`
- Node.js 20: `npm install`, `npm run build`

## 11. Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `NEUROMOVE_ROOT` | Project root path | Auto-resolved from file location |
| `NEUROMOVE_PIP_PACKAGES` | Alternate torch package path | Not set (use system torch) |

## 12. Troubleshooting

- **`ModuleNotFoundError: streamlit_option_menu`** → Install missing deps: `pip install -r requirements.txt`
- **`IndentationError` in training scripts** → Ensure you are on the latest `main` branch (syntax errors were fixed).
- **`MiniRocketPipeline.save()` fails** → Fixed: `channel_names` and `sfreq` are now set automatically in `fit()`.
- **Backend can't find engines** → Fixed: `SRC_DIR` now points to `eeg-mi-bci/src`, not the archive copy.
- **Frontend build fails** → Dependency versions in `package.json` were corrected to real, compatible releases.
