# Phase 5: Comprehensive Refactoring & Strict Validation Plan

Based on the detailed audit, we are addressing the 8 remaining critical blockers in the recommended order. This ensures scientific validity, robust execution, and accurate reporting.

## 1. Fix Cache Naming & Fail-Fast Data Handling (🔴 Blocker)
**Goal:** Unify cache generation and ensure failures are explicit.
- Create a shared `get_cache_path(dataset_name, subjects, prep_params)` function.
- Update `train_master.py` to use this single function for both checking and saving.
- Modify the data loading loop to throw an explicit `DataNotFoundError` instead of returning `None` silently when the cache misses.
- Fix `mne.concatenate_raws()` in `binary_parser.py` so it doesn't crash on `None` placeholders; explicit recording objects will track run IDs.

## 2. Real Train/Validation/Test Partitions (🔴 Blocker)
**Goal:** Prevent data leakage and establish a true unseen test set.
- Filter data to **only use Motor Imagery (MI)** runs (e.g., runs 4, 8, 12 in PhysioNet) and discard Motor Execution runs.
- Split data logically into 3 sets: Train (e.g., runs 4, 8), Validation (e.g., 20% of train runs), and Test (run 12).
- Update neural engine `.fit()` signatures to accept `X_val, y_val` instead of `X_test, y_test`.
- The final evaluation will *only* be done on `X_test` after `.fit()` finishes.

## 3. Correct Displayed Metrics & Early Stopping (🔴 / 🟠 Blocker)
**Goal:** Report mathematically sound metrics and fix early stopping logic.
- **Early Stopping (`cnn_lstm_engine.py`):** Implement standard consecutive patience (reset when val improves). Save the best state dict and restore it at the end.
- **Silent Labels:** Raise explicit `ValueError` for unknown validation labels instead of mapping to 0.
- **Dashboard / Reporting:** Remove the fake 240s timer for MiniRocket. Replace misleading "% of Model Learned" with actual Train Accuracy, Val Accuracy, and Test Accuracy.
- **MiniRocket Metrics:** Do not duplicate test accuracy as train accuracy. Calculate true train accuracy and report loss correctly.

## 4. Metadata Compatibility & Label Decoding (🟠 Blocker)
**Goal:** Ensure checkpoints are universally loadable and output correct labels.
- **Label Decoding:** Ensure `predict()` applies `classes_[np.argmax(probs)]` instead of just returning raw integer indices.
- **MiniRocket Architecture:** Clearly document that the current implementation is "MiniRocket + MLP" rather than "MiniRocket + Ridge", and align dashboard text.
- **Metadata Enforcement:** Make `sfreq` and `channel_names` mandatory (fail if `None`) during save/load, and have the loader actively apply them.

## 5. Repair Tests & Smoke Testing (🟠 Blocker)
**Goal:** Clean up the repository footprint.
- Update imports in `tests/test_pipeline.py` and any demo scripts to point to `eeg-mi-bci/src/` instead of the deleted root `src/`.
- Run a minimal smoke test to ensure everything executes.

## 6. Uncertainty Rejection (🔴 Blocker)
**Goal:** Fail gracefully on uncertain predictions.
- Implement a configurable uncertainty threshold (e.g., `max(probs) < 0.60`).
- If uncertain, output "Uncertain—please repeat the trial" in `app.py`.
- Evaluate how many incorrect predictions are caught by this threshold on the unseen test set.
