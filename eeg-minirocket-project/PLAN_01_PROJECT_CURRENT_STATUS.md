# PLAN 01 — Project Current Status

## Repository Overview
- **Root:** `D:\eeg-minirocket-project`
- **Branch:** `main`
- **Remote:** `https://github.com/shushankreddyk-lab/NEURO_MOVE`
- **Primary Python venv:** `.venv/` at root
- **Secondary backend venv:** `neuro-move-release/backend/.venv/`

## Directory Layout (tracked source)
```
eeg-minirocket-project/
├── eeg-mi-bci/                # Core BCI engines + Streamlit dashboard
│   ├── dashboard/app.py       # Main Streamlit UI (3D WebGL, tabs, KPIs)
│   ├── src/                   # MiniRocket, EEGNet, CNN-LSTM, Conformer, etc.
│   ├── models/                # (gitignored) trained weights .pth/.pkl
│   ├── artifacts/             # (gitignored) metrics, caches
│   └── requirements.txt
├── eeg-mi-bci-archive-565657a/# Duplicate/archive engine tree (user wants this as reference path)
│   ├── src/
│   ├── dashboard/
│   └── scripts/
├── neuro-move-release/        # FastAPI backend + React frontend release
│   ├── backend/
│   │   ├── run.py
│   │   ├── neuromove/         # api.py, service.py, config.py, models.py, dependencies.py
│   │   └── requirements.txt   # (missing)
│   └── frontend/
│       ├── package.json       # React 19, Vite, oxlint
│       └── src/
├── scripts/                   # 30+ training/download/fix utilities (many broken)
├── tests/                     # 3 unittest files (paths partially wrong)
├── requirements.txt           # Root Python deps
├── README.md
└── .gitignore                 # Ignores data/, models/, results/, *.pth, *.csv, etc.
```

## Key Entry Points
1. **Streamlit Dashboard:** `cd eeg-mi-bci && streamlit run dashboard/app.py`
2. **FastAPI Backend:** `cd neuro-move-release/backend && python run.py`
3. **React Frontend:** `cd neuro-move-release/frontend && npm install && npm run dev`
4. **Smoke Test:** `python scripts/smoke_test.py`

## Known Working (from previous analysis)
- `minirocket_engine.py` parses and imports; `TorchMiniRocket`, `GPUMiniRocketHead`, `MiniRocketPipeline` classes exist.
- `neuro-move-release/backend/neuromove/api.py` defines all FastAPI routes (`/health`, `/datasets`, `/preprocess`, `/train`, `/infer`, `/evaluate`).
- Frontend `App.jsx` has React Router with 5 pages.

## Known Broken / Missing
- `scripts/overtrain_targets.py` — `IndentationError`
- `scripts/training_process_snippet.py` — `IndentationError`
- `scripts/app_recovered_backup.py` — contains uncompyle6 error text, not Python
- `eeg-mi-bci-archive-565657a/dashboard/app_recovered.py` — same invalid content
- `MiniRocketPipeline.save()` always raises (requires unset `channel_names`/`sfreq`)
- `predict`/`predict_proba` lack `is_fitted` guard
- `streamlit-option-menu` and `streamlit-lottie` imported but missing from `requirements.txt`
- Hardcoded `D:\` paths in `minirocket_engine.py`, `app.py`, `config.py`
- Backend `config.py:14` points `SRC_DIR` at archive copy instead of `eeg-mi-bci/src`
- `tests/test_label_mapping.py` imports wrong `src` path
- `.gitignore` excludes all data/models/results — fresh clone cannot run
