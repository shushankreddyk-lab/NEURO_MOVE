import sys
import os
import gc
import json
import numpy as np
import datetime

# --- Force CUDA torch from D:\pip_packages (overrides system CPU torch) ---
_D_PKGS = r"D:\pip_packages"
if _D_PKGS not in sys.path:
    sys.path.insert(0, _D_PKGS)

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from binary_parser import load_local_eeg_data
from preprocessing import apply_car, apply_bandpass_filter
from minirocket_engine import MiniRocketPipeline
from advanced_eeg_engine import AdvancedEEGPipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def map_run_and_marker_to_group(run, task_type, marker_str):
    if run in [1, 2]: return -1
    if run in [3, 7, 11, 4, 8, 12]:
        if marker_str == 'T0': return -1
        if marker_str == 'T1': return 0
        if marker_str == 'T2': return 1
    if run in [5, 9, 13, 6, 10, 14]:
        if marker_str == 'T0': return -1
        if marker_str == 'T1': return 2
        if marker_str == 'T2': return 3
    return -1

def extract_subject_data(sub, dataset_path):
    runs = list(range(1, 15))
    X_list = []
    y_list = []
    
    try:
        raws, events_list, mappings = load_local_eeg_data(sub, runs, data_dir=dataset_path)
        for i, raw in enumerate(raws):
            run = runs[i]
            events = events_list[i]
            mapping = mappings[i]
            
            raw.apply_function(lambda x: x * 1e6, verbose=False)
            raw = apply_bandpass_filter(apply_car(raw), 4, 38)
            
            target_channels = ['FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4', 'C1', 'C2', 'C5', 'C6', 'CZ', 'FCZ', 'CPZ', 'F3', 'F4', 'P3', 'P4', 'O1', 'O2', 'OZ']
            from binary_parser import normalize_channel_names
            raw.rename_channels(normalize_channel_names(raw.ch_names))
            picked_channels = [ch for ch in target_channels if ch in raw.ch_names]
            if len(picked_channels) > 0:
                raw.pick_channels(picked_channels)
            
            from binary_parser import get_label_mapping
            orig_mapping = get_label_mapping(run)
            inv_orig = {v: k for k, v in orig_mapping.items()}
            
            for ev in events:
                if len(ev) == 0: continue
                onset, duration, marker_int = ev
                desc_str = mapping.get(marker_int)
                if not desc_str: continue
                marker_str = inv_orig.get(desc_str)
                if not marker_str: continue
                
                target_group = map_run_and_marker_to_group(run, "Unknown", marker_str)
                if target_group == -1: continue
                
                import mne
                tmax_adj = 4.1 - (1 / raw.info['sfreq'])
                epochs = mne.Epochs(raw, np.array([ev]), event_id={marker_str: marker_int}, tmin=0, tmax=tmax_adj, baseline=None, preload=True, verbose=False)
                X_batch = epochs.get_data(copy=False)
                
                target_samples = 656
                if X_batch.shape[2] > target_samples:
                    X_batch = X_batch[:, :, :target_samples]
                elif X_batch.shape[2] < target_samples:
                    pad_width = target_samples - X_batch.shape[2]
                    X_batch = np.pad(X_batch, ((0,0), (0,0), (0,pad_width)), mode='constant')
                
                if X_batch.shape[0] > 0:
                    X_list.append(X_batch)
                    y_list.append([target_group] * len(X_batch))
                    
        del raws
        gc.collect()
    except Exception as e:
        print(json.dumps({"type": "info", "message": f"Exception for Sub {sub}: {e}"}), flush=True)

    if len(X_list) == 0:
        return None, None
    
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all

if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(json.dumps({"type": "info", "message": f"GPU Training Mode: device={device}"}), flush=True)

    dataset_path = r"d:\eeg-minirocket-project\physionet"
    models_dir = r"d:\eeg-minirocket-project\eeg-mi-bci\models"
    os.makedirs(models_dir, exist_ok=True)
    
    subset_size = 20
    print(json.dumps({"type": "info", "message": f"Starting GPU Global Training across {subset_size} subjects..."}), flush=True)
    
    X_global = []
    y_global = []
    
    for sub in range(1, 110):
        if sub > subset_size:
            break
        if sub in [88, 89, 92, 100, 104, 106]:
            continue
            
        print(json.dumps({"type": "progress", "message": f"Extracting Subject {sub} / {subset_size}..."}), flush=True)
        X, y = extract_subject_data(sub, dataset_path)
        
        if X is not None and len(X) > 0:
            X_global.append(X.astype(np.float32))
            y_global.append(y)
    
    if len(X_global) == 0:
        print(json.dumps({"type": "error", "message": "No data extracted."}), flush=True)
        sys.exit(1)
        
    print(json.dumps({"type": "progress", "message": "Concatenating all subjects data..."}), flush=True)
    X_all = np.concatenate(X_global, axis=0)
    y_all = np.concatenate(y_global, axis=0)
    del X_global, y_global
    gc.collect()
    
    print(json.dumps({"type": "info", "message": f"Total global dataset shape: {X_all.shape}, Labels: {y_all.shape}"}), flush=True)
    
    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.1, random_state=42, stratify=y_all)
    
    # === TRAIN GPU MINIROCKET ===
    print(json.dumps({"type": "progress", "message": "Fitting GPU-Accelerated MiniRocket..."}), flush=True)
    mr_pipeline = MiniRocketPipeline(
        num_kernels=10000,
        in_channels=20,
        seq_len=656,
        hidden=512,
        dropout=0.3,
        head_epochs=30,
        head_lr=1e-3,
        batch_size=128
    )
    mr_pipeline.fit(X_train, y_train)
    
    preds_mr = mr_pipeline.predict(X_val)
    acc_mr = accuracy_score(y_val, preds_mr)
    print(json.dumps({"type": "info", "message": f"GPU MiniRocket Validation Acc: {acc_mr:.4f}"}), flush=True)
    
    # === TRAIN EEG-CONFORMER ===
    print(json.dumps({"type": "progress", "message": "Fitting Advanced EEG-Conformer on GPU..."}), flush=True)
    conformer_pipeline = AdvancedEEGPipeline(
        num_classes=4,
        channels=20,
        samples=656,
        epochs=30,
        batch_size=64,
        lr=3e-4
    )
    conformer_pipeline.fit(X_train, y_train, X_val, y_val)
    
    preds_conf = conformer_pipeline.predict(X_val)
    acc_conf = accuracy_score(y_val, preds_conf)
    print(json.dumps({"type": "info", "message": f"EEG-Conformer Validation Acc: {acc_conf:.4f}"}), flush=True)
    
    # === SAVE MODELS ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mr_path = os.path.join(models_dir, f"master_global_gpu_minirocket_{timestamp}.pth")
    conformer_path = os.path.join(models_dir, f"master_global_conformer_{timestamp}.pth")
    
    mr_pipeline.save(mr_path)
    conformer_pipeline.save(conformer_path)
    
    print(json.dumps({"type": "complete", "message": f"GPU Training Complete! Models saved:\n  MiniRocket: {mr_path}\n  Conformer: {conformer_path}"}), flush=True)
