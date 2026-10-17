# PLAN 02 — Errors and Bugs

## Critical Runtime Errors (app will crash)

| # | File | Line(s) | Issue | Impact |
|---|------|---------|-------|--------|
| E1 | `scripts/overtrain_targets.py` | 46-48 | `if not os.path.exists(mat_path):` empty body, then unindented code | `IndentationError`, script cannot run |
| E2 | `scripts/training_process_snippet.py` | 22 | `trained_models = []` indented with no enclosing block | `IndentationError` |
| E3 | `scripts/app_recovered_backup.py` | 1 | Contains literal uncompyle6 error message, not Python | Import/execution failure |
| E4 | `eeg-mi-bci-archive-565657a/dashboard/app_recovered.py` | 1 | Same invalid content | Import/execution failure |
| E5 | `eeg-mi-bci/src/minirocket_engine.py` | 244-247 | `save()` requires `channel_names` and `sfreq`, never set in `__init__`/`fit()` | `ValueError` on every `save()` |
| E6 | `eeg-mi-bci/src/minirocket_engine.py` | 208-241 | `predict`/`predict_proba` access `self.gpu_head`, `self.feat_mean`, `self.feat_std` without `is_fitted` check | `AttributeError` before `fit()` |
| E7 | `eeg-mi-bci/dashboard/app.py` | 4-5 | Imports `streamlit_option_menu`, `streamlit_lottie` | `ModuleNotFoundError` (not in requirements) |
| E8 | `neuro-move-release/backend/neuromove/config.py` | 14 | `SRC_DIR` points to `eeg-mi-bci-archive-565657a/src` (stale archive) | Backend imports wrong engine copy |

## Portability Errors

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| P1 | `eeg-mi-bci/src/minirocket_engine.py` | 6 | `D:\pip_packages` hardcoded into `sys.path` |
| P2 | `eeg-mi-bci/dashboard/app.py` | 19 | `D:\pip_packages` hardcoded into `sys.path` |
| P3 | `neuro-move-release/backend/neuromove/config.py` | 11 | `PROJECT_ROOT = Path(r"D:\eeg-minirocket-project")` |
| P4 | `neuro-move-release/backend/neuromove/config.py` | 17 | `VENV_PYTHON` hardcoded to `D:\...` |

## Deprecation / Security

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| D1 | `eeg-mi-bci/src/minirocket_engine.py` | 172, 181 | `torch.amp.GradScaler("cuda")` / `autocast("cuda")` positional string deprecated |
| D2 | `eeg-mi-bci/src/minirocket_engine.py` | 268 | `torch.load(..., weights_only=False)` security risk + warning |

## Test Issues

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| T1 | `tests/test_label_mapping.py` | 7 | Inserts `project_root/src` which does not exist; engines are in `eeg-mi-bci/src` |
| T2 | `tests/test_label_mapping.py` | 26 | Requires PhysioNet `.edf` files that are gitignored/absent |

## Project Structure Issues

| # | Issue |
|---|-------|
| S1 | `.gitignore` excludes `datasets/`, `dataset/`, `models/`, `results/`, `eeg-mi-bci/artifacts/`, `eeg-mi-bci/models/` — fresh clone has no data or models |
| S2 | Two near-identical `requirements.txt` files (root and `eeg-mi-bci/`) |
| S3 | No `backend/requirements.txt` |
| S4 | No lint/format config (`ruff`, `black`, `flake8`, pre-commit) |
| S5 | No CI workflow (GitHub Actions, etc.) |
