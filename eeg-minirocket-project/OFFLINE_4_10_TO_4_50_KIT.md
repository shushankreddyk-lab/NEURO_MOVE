# OFFLINE SURVIVAL KIT — Internet Blackout 4:10 PM to 4:50 PM IST (22 Sept 2026)
Created: 16:06 IST | Current time check: `Get-Date` = 16:04 IST

## 0. Situation
- Internet may drop ANY time after 16:10 and return ~16:50 (40 min window).
- This repo `d:\eeg-minirocket-project` is 100% usable OFFLINE. Do NOT rely on web.
- All deps already installed GLOBALLY (Python 3.13.14). All data cached locally.

## 1. USE GLOBAL PYTHON (NOT .venv) OFFLINE
`.venv/` at `d:\eeg-minirocket-project\.venv` is EMPTY (created 14:10 today, isolated, no packages).
Global `python` has everything: numpy 2.4.6, mne 1.13.2, sklearn 1.7.2, sktime 1.1.0, torch 2.13.0, streamlit 1.63.0, etc.

```powershell
# VERIFY (no internet needed):
python --version   # 3.13.14
pip list | Select-String "mne|torch|sktime|sklearn|streamlit"
# Run everything with:
python src\train.py --help
python -m unittest tests.test_pipeline -v
streamlit run demo\app.py
```

DO NOT run `pip install` during blackout. If you must, use:
```powershell
pip install --no-index --find-links=C:\pip-cache <pkg>  # will fail if not cached — avoid
```

## 2. DATA IS LOCAL — NO DOWNLOAD NEEDED
- PhysioNet cache: `d:\eeg-minirocket-project\physionet\` = 3.4 GB, 116 entries, S001..S109 EDF + .event files. Verified 22-09-2026.
- Models (pretrained, no training needed offline):
  - `models\minirocket_physionet_s01.joblib` (633 KB)
  - `models\cnn_lstm_physionet_s01.pt` (1.9 MB)
  - `models\minirocket_bci_iv_2a_s01.joblib` (147 KB)
  - `models\cnn_lstm_bci_iv_2a_s01.pt` (1.6 MB)
  - `models\sample_test_data_physionet.npz` (2.4 MB) + `sample_test_data_bci_iv_2a.npz` (2.6 MB)
- Results already computed: `results\metrics\benchmark_summary_*.csv`, `results\plots\*.png`

`src\data_loader.py` defaults that keep you offline:
- `load_physionet_subject(..., use_synthetic_fallback=False)` -> forces LOCAL EDF, will raise if file missing (good — you know instantly).
- `load_bci_iv_2a_subject(..., use_synthetic_fallback=True)` -> falls back to SYNTHETIC if GDF missing (no internet).
- `generate_synthetic_eeg()` -> 100% offline, use for fast tests.

## 3. COMMANDS THAT WORK 100% OFFLINE (copy-paste)

### A. Fastest sanity check (<30 sec, synthetic only, no disk I/O):
```powershell
cd d:\eeg-minirocket-project
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1; $env:HF_DATASETS_OFFLINE=1
python -c "from src.data_loader import generate_synthetic_eeg; from src.preprocessing import preprocess_eeg_dataset; from src.minirocket_pipeline import MiniRocketPipeline; X,y,ch=generate_synthetic_eeg(n_epochs=20,n_channels=16,n_times=256); Xp,yp=preprocess_eeg_dataset(X,y,sfreq=128.0); m=MiniRocketPipeline(num_kernels=200); m.fit(Xp,yp); print('OFFLINE OK acc=',m.score(Xp,yp))"
```

### B. Full unit tests (2-4 min, synthetic + tiny torch train, offline):
```powershell
cd d:\eeg-minirocket-project
$env:HF_HUB_OFFLINE=1; $env:TRANSFORMERS_OFFLINE=1
python -m unittest tests.test_pipeline -v
```

### C. Train offline on LOCAL PhysioNet (no download):
```powershell
cd d:\eeg-minirocket-project
# 1 subject, 500 kernels, 2 epochs = ~1-2 min smoke test:
python src\train.py --dataset physionet --num_subjects 1 --num_kernels 500 --epochs 2 --batch_size 32
# Full repro (10 subjects, 2000 kernels, 15 epochs) = run AFTER 16:50 if you have time:
python src\train.py --dataset physionet --num_subjects 10 --num_kernels 2000 --epochs 15
# BCI synthetic (no GDF files needed):
python src\train.py --dataset bci_iv_2a --num_subjects 3 --num_kernels 500 --epochs 2
```

### D. Evaluate offline (uses local models + npz, no internet):
```powershell
cd d:\eeg-minirocket-project
python src\evaluate.py --dataset physionet --model_dir models --results_dir results
python src\evaluate.py --dataset bci_iv_2a --model_dir models --results_dir results
```

### E. Demo offline:
```powershell
cd d:\eeg-minirocket-project
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
streamlit run demo\app.py --server.headless true --global.disableWatchdogWarning true
# Open http://localhost:8501
# NOTE: Google Fonts @import in app.py will fail offline — UI falls back to system font, functionality UNAFFECTED.
```

### F. Notebook offline:
```powershell
cd d:\eeg-minirocket-project
jupyter notebook notebooks\exploration.ipynb  # if jupyter installed, else use VS Code notebook (already cached kernel)
# All cells use generate_synthetic_eeg or local EDF — set use_synthetic_fallback=True if EDF missing.
```

## 4. WHAT WILL BREAK OFFLINE (avoid 16:10-16:50)
- `pip install <new pkg>`, `pip download`, `mne.datasets.*.fetch_*`, `huggingface_hub` downloads, `pooch` downloads.
- `streamlit` Google Fonts, `plotly` CDN, GitHub push/pull, VS Code Extensions install, Copilot/Cloud LLMs.
- Fix: set these BEFORE 16:10:
```powershell
$env:HF_HUB_OFFLINE=1
$env:TRANSFORMERS_OFFLINE=1
$env:HF_DATASETS_OFFLINE=1
$env:PYTHONNOUSERSITE=0
```

## 5. 40-MIN OFFLINE TODO (no internet needed, high value)
1. [10 min] Run B + C-smoke + D, screenshot `results\metrics\benchmark_summary_physionet.csv`
2. [10 min] Edit `notebooks\exploration.ipynb` — add confusion matrix cell from `src\evaluate.py:plot_confusion_matrices`
3. [10 min] Test `demo\app.py` latency display vs `MiniRocketPipeline.benchmark_latency()` (target ~0.6ms)
4. [10 min] Write `results\OFFLINE_NOTES.md` — copy accuracies for report (MiniRocket ~98.6% PhysioNet / ~92.6% BCI per train.py:199)

## 6. AFTER 16:50 (when net returns)
```powershell
cd d:\eeg-minirocket-project
git status; git diff --stat
pip list --outdated
streamlit run demo\app.py  # fonts will reload
```

## 7. Emergency contacts / files
- Base paper text: `base_paper_text.txt` (97 KB, local)
- PhysioNet mapping: `eeg-mi-bci\physionet_run_mapping.csv` (120 KB)
- Logs: `eeg-mi-bci\streamlit_out.log`, `eeg-mi-bci\logs\`
- If VS Code AI offline fails, use local python help: `python -c "help('src.minirocket_pipeline.MiniRocketPipeline.fit')"`

Good luck — you are fully covered for 4:10-4:50. Do B + C-smoke NOW before 4:10 to confirm.
