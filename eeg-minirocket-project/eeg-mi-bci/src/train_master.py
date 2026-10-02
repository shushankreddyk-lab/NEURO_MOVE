import os
import sys

# --- Force CUDA torch from D:\pip_packages (overrides system CPU torch) ---
_D_PKGS = r"D:\pip_packages"
if _D_PKGS not in sys.path:
    sys.path.insert(0, _D_PKGS)

import numpy as np
import json
import gc
import shutil
import datetime
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from binary_parser import load_local_eeg_data
from preprocessing import apply_car, apply_bandpass_filter, spatial_channel_augmentation, epoch_and_segment
from minirocket_engine import MiniRocketPipeline
from advanced_eeg_engine import AdvancedEEGPipeline
from eegnet_engine import EEGNet_Pipeline
from convnets_engine import ConvNet_Pipeline

def get_cache_path(mode, group_id, dataset_path, sub_start, sub_end, data_dir):
    import hashlib
    import os
    sub_str = f"subs{sub_start}to{sub_end}"
    
    dataset_name = dataset_path.lower().replace(" ", "_").replace("\\", "_").replace("/", "_")
    if "2a" in dataset_path.lower():
        dataset_name = "bci2a"
    elif "physionet" in dataset_path.lower():
        dataset_name = "physionetmi"
    
    # Include dataset identity and preprocessing settings (e.g. v3)
    config_str = f"{mode}_{group_id}_{sub_str}_{dataset_name}_v3"
    hash_key = hashlib.md5(config_str.encode('utf-8')).hexdigest()[:8]
    
    fname = f"data_{dataset_name}_{sub_str}_{hash_key}.npz"
    return os.path.join(data_dir, fname)


def map_run_and_marker_to_group(run, task_type, marker_str):
    # Returns the exact integer class label (0 to 3) expected by the UI
    if run in [1, 2]: return -1 # Eyes Open/Closed (Baseline)
    # Group Real and MI together for fists
    if run in [3, 7, 11, 4, 8, 12]:
        if marker_str == 'T0': return -1 # Rest
        if marker_str == 'T1': return 0 # Left Fist
        if marker_str == 'T2': return 1 # Right Fist
    # Group Real and MI together for fists/feet
    if run in [5, 9, 13, 6, 10, 14]:
        if marker_str == 'T0': return -1 # Rest
        if marker_str == 'T1': return 2 # Both Fists
        if marker_str == 'T2': return 3 # Both Feet
    return -1 # Fallback to Rest

from cnn_lstm_engine import CNN_LSTM_Pipeline

def archive_old_models(models_dir):
    files_to_archive = [f for f in os.listdir(models_dir) if f.endswith('.pkl') or f.endswith('.pth')]
    if files_to_archive:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(models_dir, 'archive', f'session_{timestamp}')
        os.makedirs(archive_dir, exist_ok=True)
        for f in files_to_archive:
            shutil.move(os.path.join(models_dir, f), os.path.join(archive_dir, f))
        print(json.dumps({"type": "info", "message": f"Archived {len(files_to_archive)} old models."}), flush=True)

def extract_and_save_data(mode, group_id, output_dir, dataset_path, sub_start=1, sub_end=1):
    os.makedirs(output_dir, exist_ok=True)
    sub_str = f"subs{sub_start}to{sub_end}"
    
    if "2a" in dataset_path.lower():
        msg = f"Extracting BCI 2a data for subjects {sub_start} to {sub_end}..."
        print(json.dumps({"type": "progress", "message": msg}), flush=True)
        
        # We need to import our new loader
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
        from dataset_2a_loader import load_bci_2a_data
        
        subject_range = list(range(sub_start, sub_end + 1))
        # Keep 160Hz for consistency with standard architecture (or use 250Hz, but let's downsample for speed)
        X_all, y_all = load_bci_2a_data(dataset_path, subject_range, resample_freq=160.0)
        
        if len(X_all) == 0:
            return None, None
            
        # Target samples matching PhysioNet
        target_samples = 656
        if X_all.shape[2] > target_samples:
            X_all = X_all[:, :, :target_samples]
        if X_all.shape[2] < target_samples:
            pad_width = target_samples - X_all.shape[2]
            X_all = np.pad(X_all, ((0,0), (0,0), (0,pad_width)), mode='constant')
            
        data_path = get_cache_path(mode, group_id, dataset_path, sub_start, sub_end, output_dir)
        np.savez_compressed(data_path, X=X_all, y=y_all)
        return X_all.shape, y_all.shape

    elif "dreamer" in dataset_path.lower():
        msg = f"Extracting DREAMER data for subjects {sub_start} to {sub_end}..."
        print(json.dumps({"type": "progress", "message": msg}), flush=True)
        
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
        from dreamer_loader import DREAMERLoader
        
        if dataset_path.endswith('.mat') and os.path.isfile(dataset_path):
            mat_path = dataset_path
        else:
            mat_path = os.path.join(dataset_path, "DREAMER.mat")
        loader = DREAMERLoader(mat_path, window_size_sec=1.0, overlap=0.5)
        X_all, y_all = loader.load_data()
        
        if len(X_all) == 0:
            return None, None
            
        data_path = get_cache_path(mode, group_id, dataset_path, sub_start, sub_end, output_dir)
        np.savez_compressed(data_path, X=X_all, y=y_all)
        return X_all.shape, y_all.shape
        
    elif "high-gamma" in dataset_path.lower():
        msg = f"Extracting High-Gamma data for subjects {sub_start} to {sub_end}..."
        print(json.dumps({"type": "progress", "message": msg}), flush=True)
        
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
        from high_gamma_loader import load_high_gamma_data
        
        subject_range = list(range(sub_start, sub_end + 1))
        X_all, y_all = load_high_gamma_data(dataset_path, subject_range, resample_freq=160.0)
        
        if len(X_all) == 0:
            return None, None
            
        data_path = get_cache_path(mode, group_id, dataset_path, sub_start, sub_end, output_dir)
        np.savez_compressed(data_path, X=X_all, y=y_all)
        return X_all.shape, y_all.shape

    elif "way" in dataset_path.lower() or "grasp" in dataset_path.lower():
        msg = f"Extracting WAY-EEG-GAL data for subjects {sub_start} to {sub_end}..."
        print(json.dumps({"type": "progress", "message": msg}), flush=True)
        
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
        from way_loader import load_way_data
        
        subject_range = list(range(sub_start, sub_end + 1))
        X_all, y_all = load_way_data(dataset_path, subject_range, resample_freq=160.0)
        
        if len(X_all) == 0:
            return None, None
            
        data_path = get_cache_path(mode, group_id, dataset_path, sub_start, sub_end, output_dir)
        np.savez_compressed(data_path, X=X_all, y=y_all)
        return X_all.shape, y_all.shape

    # Existing Physionet logic below
    # Determine runs to load based on mode
    # For master or OVR, we theoretically need ALL runs (1-14) to get all 10 groups
    runs = list(range(1, 15))
    
    X_list = []
    y_list = []
    meta_list = []
    
    msg = f"Extracting data for {'10-Class Master' if mode == 'master' else f'OVR Group {group_id}'}..."
    print(json.dumps({"type": "progress", "message": msg}), flush=True)
    
    # Just load first 1 subject for speed in this demo, otherwise RAM explodes
    # In full production, we'd use a generator or load in batches
    subject_range = range(sub_start, sub_end + 1) 
    
    for sub in subject_range:
        if sub in [88, 89, 92, 100, 104, 106]:
            continue
        try:
            raws, events_list, mappings = load_local_eeg_data(sub, runs, data_dir=dataset_path)
            for i, raw in enumerate(raws):
                run = runs[i]
                if raw is None:
                    continue
                events = events_list[i]
                mapping = mappings[i]
                
                raw = apply_bandpass_filter(apply_car(raw), 4, 38)
                
                # Determine task_type based on run
                if run in [1, 2]: task_type = "Baseline"
                elif run in [3, 7, 11]: task_type = "Motor Execution (Fists)"
                elif run in [4, 8, 12]: task_type = "Motor Imagery (Fists)"
                elif run in [5, 9, 13]: task_type = "Motor Execution (Fists/Feet)"
                elif run in [6, 10, 14]: task_type = "Motor Imagery (Fists/Feet)"
                else: task_type = "Unknown"
                from binary_parser import get_label_mapping
                orig_mapping = get_label_mapping(run)
                inv_orig = {v: k for k, v in orig_mapping.items()}
                
                valid_events = []
                valid_event_ids = {} 
                event_to_label = {} 

                for ev in events:
                    if len(ev) == 0: continue
                    onset, duration, marker_int = ev
                    desc_str = mapping.get(marker_int)
                    if not desc_str: continue
                    marker_str = inv_orig.get(desc_str)
                    if not marker_str: continue
                    
                    target_group = map_run_and_marker_to_group(run, task_type, marker_str)
                    
                    # Remove 'rest' and 'baseline' part while preprocessing and training
                    if target_group == -1:
                        continue
                    
                    if mode == "ovr":
                        lbl = 1 if str(target_group) == str(group_id) else 0
                    else:
                        lbl = target_group # 0 to 9 directly
                        
                    valid_events.append(ev)
                    valid_event_ids[str(marker_int)] = marker_int
                    event_to_label[marker_int] = lbl
                    
                if not valid_events:
                    continue
                    
                # Epoching manually for ALL valid events at once (drastically faster)
                import mne
                tmax_adj = 4.1 - (1 / raw.info['sfreq'])
                epochs = mne.Epochs(raw, np.array(valid_events), event_id=valid_event_ids, tmin=0, tmax=tmax_adj, baseline=None, preload=True, verbose=False)
                X_batch = epochs.get_data(copy=False)
                
                target_samples = 656
                if X_batch.shape[2] > target_samples:
                    X_batch = X_batch[:, :, :target_samples]
                elif X_batch.shape[2] < target_samples:
                    pad_width = target_samples - X_batch.shape[2]
                    X_batch = np.pad(X_batch, ((0,0), (0,0), (0,pad_width)), mode='constant')
                
                if X_batch.shape[0] > 0:
                    y_batch = [event_to_label[ev[2]] for ev in epochs.events]
                    meta_batch = []
                    for ev_idx, ev in enumerate(epochs.events):
                        marker_int = ev[2]
                        lbl = event_to_label[marker_int]
                        desc_str = mapping.get(marker_int)
                        marker_str = inv_orig.get(desc_str)
                        meta_batch.append({
                            "dataset_id": "PhysionetMI",
                            "subject_id": sub,
                            "run_id": run,
                            "trial_id": ev_idx,
                            "class_label": lbl,
                            "marker": marker_str,
                            "task": task_type
                        })
                    X_list.append(X_batch)
                    y_list.extend(y_batch)
                    meta_list.extend(meta_batch)
                else:
                    print("X_batch was empty!", flush=True)
                        
            del raws
            gc.collect()
        except Exception as e:
            print(f"Exception: {e}", flush=True)


    if len(X_list) == 0:
        return None, None
        
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.asarray(y_list, dtype=np.int64)
    meta_all = np.array(meta_list, dtype=object)
    
    if X_all.shape[0] != y_all.shape[0]:
        print(f"Error: Mismatched X ({X_all.shape[0]}) and y ({y_all.shape[0]}) sizes.", file=sys.stderr)
        sys.exit(1)
    
    data_path = get_cache_path(mode, group_id, dataset_path, sub_start, sub_end, output_dir)
    np.savez_compressed(data_path, X=X_all, y=y_all, meta=meta_all)
    
    return X_all.shape, y_all.shape

def train_models(mode, group_id, data_dir, models_dir, dataset_path="", model_name="MiniRocket", epochs=10, lr=1e-3, kernels=10000, train_split=0.8, sub_start=1, sub_end=1):
    sub_str = f"subs{sub_start}to{sub_end}"
    
    data_path = get_cache_path(mode, group_id, dataset_path, sub_start, sub_end, data_dir)
    model_prefix = "bci2a" if "2a" in dataset_path.lower() else ("master" if mode == "master" else f"ovr_group{group_id}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data not found at {data_path}. Please extract features first.")
        
        
    # Prefer pre-saved float32 version if available (avoids 1.33GB float64 allocation)
    f32_path = data_path.replace('.npz', '_f32.npz')
    if os.path.exists(f32_path):
        loaded = np.load(f32_path)
        X = loaded['X']  # already float32
        y = np.array(loaded['y'])
    else:
        # mmap + cast: avoids holding float64 and float32 in RAM simultaneously
        loaded = np.load(data_path, mmap_mode='r')
        X = np.array(loaded['X'], dtype=np.float32)
        y = np.array(loaded['y'])
    
    # PREVENT TEMPORAL LEAKAGE: Chronological Stratified Split
    meta = loaded.get('meta')
    
    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    
    if meta is not None and meta.ndim > 0 and len(meta) == len(y):
        print(json.dumps({"type": "info", "message": "Using metadata-driven strict run partitioning..."}), flush=True)
        for subject_id in np.unique([m['subject_id'] for m in meta]):
            sub_mask = np.array([m['subject_id'] == subject_id for m in meta])
            sub_X = X[sub_mask]
            sub_y = y[sub_mask]
            sub_meta = meta[sub_mask]
            
            sub_runs = np.unique([m['run_id'] for m in sub_meta])
            sub_runs = np.sort(sub_runs)
            
            if len(sub_runs) >= 3:
                test_runs = [sub_runs[-1]]
                val_runs = [sub_runs[-2]]
                train_runs = sub_runs[:-2]
            elif len(sub_runs) == 2:
                test_runs = [sub_runs[-1]]
                val_runs = [] 
                train_runs = [sub_runs[0]]
            else:
                train_runs = sub_runs
                val_runs, test_runs = [], []
                
            for i, m in enumerate(sub_meta):
                if m['run_id'] in test_runs:
                    X_test.append(sub_X[i])
                    y_test.append(sub_y[i])
                elif m['run_id'] in val_runs:
                    X_val.append(sub_X[i])
                    y_val.append(sub_y[i])
                else:
                    X_train.append(sub_X[i])
                    y_train.append(sub_y[i])
    else:
        print(json.dumps({"type": "info", "message": "No metadata found. Using temporal slice..."}), flush=True)
        if y.ndim > 1 or np.issubdtype(y.dtype, np.floating):
            # Regression - chronological slice to prevent leakage
            val_split = int(len(y) * 0.8)
            test_split = int(len(y) * 0.9)
            X_train, y_train = X[:val_split], y[:val_split]
            X_val, y_val = X[val_split:test_split], y[val_split:test_split]
            X_test, y_test = X[test_split:], y[test_split:]
        else:
            # Classification - stratified temporal slice
            X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list = [], [], [], [], [], []
            for cls in np.unique(y):
                idx = np.where(y == cls)[0]
                val_split = int(len(idx) * 0.8)
                test_split = int(len(idx) * 0.9)
                X_train_list.append(X[idx[:val_split]])
                y_train_list.append(y[idx[:val_split]])
                X_val_list.append(X[idx[val_split:test_split]])
                y_val_list.append(y[idx[val_split:test_split]])
                X_test_list.append(X[idx[test_split:]])
                y_test_list.append(y[idx[test_split:]])
            
            X_train = np.concatenate(X_train_list, axis=0) if X_train_list else np.empty((0, *X.shape[1:]))
            y_train = np.concatenate(y_train_list, axis=0) if y_train_list else np.empty((0,))
            X_val = np.concatenate(X_val_list, axis=0) if X_val_list else np.empty((0, *X.shape[1:]))
            y_val = np.concatenate(y_val_list, axis=0) if y_val_list else np.empty((0,))
            X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.empty((0, *X.shape[1:]))
            y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.empty((0,))
    
    if len(X_val) == 0 and len(X_train) > 0:
        new_X_train, new_y_train, new_X_val, new_y_val = [], [], [], []
        for cls in np.unique(y_train):
            idx = np.where(y_train == cls)[0]
            split = int(len(idx) * 0.8)
            new_X_train.append(X_train[idx[:split]])
            new_y_train.append(y_train[idx[:split]])
            new_X_val.append(X_train[idx[split:]])
            new_y_val.append(y_train[idx[split:]])
        X_train = np.concatenate(new_X_train, axis=0)
        y_train = np.concatenate(new_y_train, axis=0)
        X_val = np.concatenate(new_X_val, axis=0)
        y_val = np.concatenate(new_y_val, axis=0)
        
    if len(X_test) == 0:
        X_test, y_test = X_val.copy(), y_val.copy()
    
    # Shuffle internally for model training stability
    train_idx = np.random.permutation(len(X_train))
    X_train, y_train = X_train[train_idx], y_train[train_idx]
    
    val_idx = np.random.permutation(len(X_val))
    X_val, y_val = X_val[val_idx], y_val[val_idx]
    
    test_idx = np.random.permutation(len(X_test))
    X_test, y_test = X_test[test_idx], y_test[test_idx]
    
    try:
        from dataset_2a_loader import augment_data
        print(json.dumps({"type": "info", "message": f"Augmenting training data... Original shape: {X_train.shape}"}), flush=True)
        X_train, y_train = augment_data(X_train, y_train)
        print(json.dumps({"type": "info", "message": f"Augmented shape: {X_train.shape}"}), flush=True)
    except ImportError:
        pass
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_str = f"subs{sub_start}to{sub_end}"
    
    is_regression = (y.ndim > 1 or np.issubdtype(y.dtype, np.floating))
    num_cls = y.shape[1] if is_regression else len(np.unique(y))
    task_type = "regression" if is_regression else "classification"
    
    # --- MODEL EXECUTION SWITCH ---
    if model_name == "MiniRocket":
        # Train MiniRocket (GPU-accelerated)
        print(json.dumps({"type": "progress", "message": "Training GPU-Accelerated MiniRocket..."}), flush=True)
        mr_pipeline = MiniRocketPipeline(num_kernels=kernels, in_channels=X.shape[1], seq_len=X.shape[2], head_epochs=epochs, head_lr=lr)
        mr_pipeline.sfreq = 160.0
        mr_pipeline.channel_names = [f"EEG_{i}" for i in range(X.shape[1])]
        
        try:
            if args.finetune_model:
                mr_pipeline.load(args.finetune_model)
            mr_pipeline.fit(X_train, y_train)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
            sys.exit(1)
            
        mr_pipeline.save(os.path.join(models_dir, f"{model_prefix}_gpu_minirocket_{sub_str}_{timestamp}.pth"))

    elif model_name == "CNN-LSTM":
        # Train CNN-LSTM
        print(json.dumps({"type": "progress", "message": "Training CNN-LSTM Baseline..."}), flush=True)
        print(json.dumps({"type": "reset_chart"}), flush=True)
        
        cnn_lstm_pipeline = CNN_LSTM_Pipeline(
            num_classes=num_cls,
            channels=X.shape[1],
            samples=X.shape[2],
            epochs=epochs,
            lr=lr,
            batch_size=64,
            task_type=task_type
        )
        
        try:
            if args.finetune_model:
                cnn_lstm_pipeline.load(args.finetune_model)
            cnn_lstm_pipeline.fit(X_train, y_train, X_val, y_val)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
            sys.exit(1)
    
        cnn_lstm_pipeline.save(os.path.join(models_dir, f"{model_prefix}_cnn_lstm_{sub_str}_{timestamp}.pth"))

    elif model_name == "Advanced Transformer":
        # Train Pure Transformer (Conformer)
        print(json.dumps({"type": "progress", "message": "Training Advanced Transformer (Conformer)..."}), flush=True)
        print(json.dumps({"type": "reset_chart"}), flush=True)
        
        conformer_pipeline = AdvancedEEGPipeline(
            num_classes=num_cls,
            channels=X.shape[1],
            samples=X.shape[2],
            epochs=epochs,
            lr=lr,
            batch_size=64,
            task_type=task_type
        )
        
        try:
            if args.finetune_model:
                conformer_pipeline.load(args.finetune_model)
            conformer_pipeline.fit(X_train, y_train, X_val, y_val)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
            sys.exit(1)
    
        conformer_pipeline.save(os.path.join(models_dir, f"{model_prefix}_conformer_{sub_str}_{timestamp}.pth"))

    elif model_name == "EEGNet":
        print(json.dumps({"type": "progress", "message": "Training EEGNet..."}), flush=True)
        print(json.dumps({"type": "reset_chart"}), flush=True)
        
        eegnet_pipeline = EEGNet_Pipeline(
            num_classes=num_cls,
            channels=X.shape[1],
            samples=X.shape[2],
            epochs=epochs,
            lr=lr,
            batch_size=64,
            task_type=task_type
        )
        
        try:
            if args.finetune_model:
                eegnet_pipeline.load(args.finetune_model)
            eegnet_pipeline.fit(X_train, y_train, X_val, y_val)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
            sys.exit(1)
    
        eegnet_pipeline.save(os.path.join(models_dir, f"{model_prefix}_eegnet_{sub_str}_{timestamp}.pth"))

    elif model_name == "Shallow ConvNet":
        print(json.dumps({"type": "progress", "message": "Training Shallow ConvNet..."}), flush=True)
        print(json.dumps({"type": "reset_chart"}), flush=True)
        shallow_pipeline = ConvNet_Pipeline(arch="shallow", num_classes=num_cls, channels=X.shape[1], samples=X.shape[2], epochs=epochs, lr=lr)
        try:
            if args.finetune_model:
                shallow_pipeline.load(args.finetune_model)
            shallow_pipeline.fit(X_train, y_train, X_val, y_val)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
            sys.exit(1)
        shallow_pipeline.save(os.path.join(models_dir, f"{model_prefix}_shallow_{sub_str}_{timestamp}.pth"))

    # Removed Deep ConvNet and CSP+LDA as requested

    # Compute final metrics for the completion message
    final_acc = 0.0
    val_acc = 0.0
    latency_ms = 0.0
    try:
        import time
        pipeline_to_eval = None
        if model_name == "MiniRocket": pipeline_to_eval = mr_pipeline
        elif model_name == "CNN-LSTM": pipeline_to_eval = cnn_lstm_pipeline
        elif model_name == "Advanced Transformer": pipeline_to_eval = conformer_pipeline
        elif model_name == "EEGNet": pipeline_to_eval = eegnet_pipeline
        elif model_name == "Shallow ConvNet": pipeline_to_eval = shallow_pipeline
        
        if pipeline_to_eval:
            # Measure latency on 1 sample to simulate real-time BCI latency
            start_t = time.time()
            _ = pipeline_to_eval.predict(X_test[:1])
            end_t = time.time()
            latency_ms = (end_t - start_t) * 1000.0
            
            # Measure final validation accuracy
            y_pred_all = pipeline_to_eval.predict(X_test)
            final_acc = float(accuracy_score(y_test, y_pred_all))
            
            y_pred_val = pipeline_to_eval.predict(X_val)
            val_acc = float(accuracy_score(y_val, y_pred_val))
            
            if model_name == "MiniRocket":
                y_pred_train = pipeline_to_eval.predict(X_train)
                train_acc = float(accuracy_score(y_train, y_pred_train))
                print(json.dumps({
                    "type": "chart_update",
                    "epoch": epochs,
                    "train_loss": 0.0,
                    "train_acc": train_acc,
                    "val_loss": 0.0,
                    "val_acc": val_acc
                }), flush=True)
    except Exception as e:
        print(f"Error calculating metrics: {e}", file=sys.stderr)
        
    print(json.dumps({
        "type": "complete", 
        "message": f"Training completed successfully for {model_name} on {model_prefix}!",
        "final_val_acc": final_acc,
        "latency_ms": latency_ms
    }), flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["master", "ovr"])
    parser.add_argument("--model", type=str, default="MiniRocket", help="The architecture to train.")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kernels", type=int, default=10000)
    parser.add_argument("--partition", type=int, default=80)
    parser.add_argument("--finetune_model", type=str, default="", help="Path to pre-trained model for subject fine-tuning")
    parser.add_argument("--sub_start", type=int, default=1)
    parser.add_argument("--sub_end", type=int, default=1)
    
    args = parser.parse_args()
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    data_path = get_cache_path(args.mode, args.group, args.dataset, args.sub_start, args.sub_end, data_dir)
    
    if not os.path.exists(data_path):
        unified_datasets = ["BNCI2014_001", "PhysionetMI", "HighGamma", "KayaFingers", "WayEEGGAL"]
        
        # Strip trailing slash or path elements if user passed a path instead of an ID
        ds_id = args.dataset
        ds_lower = args.dataset.lower()
        if "kaya" in ds_lower: ds_id = "KayaFingers"
        elif "way" in ds_lower or "grasp" in ds_lower: ds_id = "WayEEGGAL"
        elif "high-gamma" in ds_lower or "nemar" in ds_lower or "nm000172" in ds_lower: ds_id = "HighGamma"
        elif "bci" in ds_lower or "2a" in ds_lower: ds_id = "BNCI2014_001"
        elif "physionet" in ds_lower: ds_id = "PhysionetMI"
        
        if ds_id in unified_datasets:
            sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
            from dataset_loader_all import load_dataset
            print(json.dumps({"type": "progress", "message": f"Extracting {ds_id} data using unified loader..."}), flush=True)
            
            try:
                X_train, y_train, X_test, y_test = load_dataset(ds_id, subject_id=args.sub_start)
                X_all = np.concatenate([X_train, X_test], axis=0)
                y_all = np.concatenate([y_train, y_test], axis=0)
                np.savez_compressed(data_path, X=X_all, y=y_all)
            except Exception as e:
                print(json.dumps({"type": "error", "message": f"Failed to extract {ds_id}: {str(e)}"}), flush=True)
                sys.exit(1)
        else:
            shapeX, shapeY = extract_and_save_data(args.mode, args.group, data_dir, args.dataset, sub_start=args.sub_start, sub_end=args.sub_end)
            if shapeX is None:
                print(json.dumps({"type": "error", "message": "No data extracted."}), flush=True)
                sys.exit(1)
    if args.model == "all":
        architectures = ["MiniRocket", "CNN-LSTM", "Advanced Transformer", "EEGNet", "Shallow ConvNet"]
    else:
        architectures = [args.model]
        
    for arch in architectures:
        train_models(
            mode=args.mode, 
            group_id=args.group, 
            data_dir=data_dir, 
            models_dir=models_dir,
            dataset_path=args.dataset,
            model_name=arch,
            epochs=args.epochs,
            lr=args.lr,
            kernels=args.kernels,
            train_split=args.partition / 100.0,
            sub_start=args.sub_start,
            sub_end=args.sub_end
        )
