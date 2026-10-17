# PLAN 03 — Upgrade Roadmap

## Phase 1 — Stabilize (must fix before anything runs)
1. Fix or delete `scripts/overtrain_targets.py` (`IndentationError`)
2. Fix `scripts/training_process_snippet.py` (`IndentationError`)
3. Delete or replace `scripts/app_recovered_backup.py` and `eeg-mi-bci-archive-565657a/dashboard/app_recovered.py`
4. Fix `MiniRocketPipeline.save()` — set `channel_names`/`sfreq` in `fit()`, make `save()` tolerant
5. Add `is_fitted` guard to `predict`/`predict_proba`
6. Add `streamlit-option-menu` and `streamlit-lottie` to `requirements.txt` (root + `eeg-mi-bci/`)
7. Remove hardcoded `D:\` paths; use `pathlib` + env var `NEUROMOVE_ROOT`
8. Fix backend `config.py` `SRC_DIR` → `eeg-mi-bci/src`
9. Fix `torch.amp` deprecation and `torch.load(weights_only=True)`

## Phase 2 — Backend & Frontend Hardening
10. Create `neuro-move-release/backend/requirements.txt` (fastapi, uvicorn, pydantic, etc.)
11. Consolidate root and `eeg-mi-bci/` `requirements.txt` into one canonical file
12. Fix frontend dependency versions (React 19 + Vite + `@vitejs/plugin-react` compatibility)
13. Add frontend `README` with `npm install` / `npm run dev` steps
14. Fix `tests/test_label_mapping.py` import path; add data-availability skip logic

## Phase 3 — Tooling & CI
15. Add `ruff` (lint + format) and `.pre-commit-config.yaml`
16. Add `pyproject.toml` with `pytest` config
17. Add GitHub Actions workflow: install deps → `ruff check` → `pytest` → smoke test
18. Create `docs/setup.md` with environment, data placement, and run instructions

## Phase 4 — Data & Model Management
19. Un-ignore a small sample dataset or add `data/sample/` with synthetic EEG
20. Add `models/.gitkeep` and document where to place `.pth`/`.pkl` weights
21. Consider Git LFS for model weights if they must be versioned
22. Centralize hyperparameters into `config.yaml` or `config.py`

## Phase 5 — Archive-565657 Frontend Representation (per user request)
23. Build HTML representation of `eeg-mi-bci-archive-565657a/` in frontend
24. Map archive `src/` structure to frontend pages/components
25. Ensure archive dashboard visuals render in React/Three.js frontend

## Definition of Done per Upgrade
- [ ] Code change made
- [ ] File(s) linted with `ruff`
- [ ] Relevant tests pass (`pytest`)
- [ ] Commit with descriptive message
- [ ] Push to `origin main`
