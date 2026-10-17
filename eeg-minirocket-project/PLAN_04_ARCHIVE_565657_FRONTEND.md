# PLAN 04 — Archive-565657 Frontend HTML Representation

## Goal
The frontend (`neuro-move-release/frontend/`) must represent the archive tree `eeg-mi-bci-archive-565657a/` as HTML/pages, since the user identified `archive-565657/src` as the main path for status checking.

## Archive Structure to Represent
```
eeg-mi-bci-archive-565657a/
├── src/              # Engines (miniRocket, EEGNet, etc.)
├── dashboard/        # Streamlit app.py + 3D components
├── scripts/          # Training/download utilities
├── artifacts/        # Metrics and caches
├── figures/          # Architecture diagrams
├── reports/          # HTML/PDF/MD reports
└── requirements.txt
```

## Frontend Pages to Add / Modify
1. **Dashboard** — Archive status overview (src files count, last modified, key engines)
2. **Archive Explorer** — File tree view of `eeg-mi-bci-archive-565657a/src/` and `dashboard/`
3. **Engine Status** — Per-engine availability (MiniRocket, EEGNet, CNN-LSTM, Conformer)
4. **Reports Viewer** — Embed/archive HTML reports from `reports/`
5. **Figures Viewer** — Image gallery from `figures/`

## Implementation Steps
1. Add route `/archive` in `frontend/src/App.jsx`
2. Create `frontend/src/pages/Archive.jsx`
3. Create `frontend/src/components/ArchiveFileTree.jsx`
4. Create `frontend/src/components/EngineStatusCard.jsx`
5. Add API endpoints in backend to list archive files (`/archive/files`, `/archive/engines`)
6. Serve static archive HTML/reports via backend `/archive/reports/*`

## Backend Additions
- `GET /archive/tree` — returns nested file/folder structure of archive
- `GET /archive/engines` — returns list of engine modules found in `archive/src/`
- `GET /archive/report/{name}` — serves archived HTML report

## Success Criteria
- Frontend navigates to archive section without errors
- Archive `src/` file tree renders
- Archived reports display correctly
- Backend health reports archive directory existence
