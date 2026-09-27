import sys
import os
import gc
import json
import numpy as np
import datetime
import shutil
import threading
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from binary_parser import load_local_eeg_data
from preprocessing import apply_car, apply_bandpass_filter
from minirocket_engine import MiniRocketPipeline
from cnn_lstm_engine import CNN_LSTM_Pipeline

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
            
            raw = apply_bandpass_filter(apply_car(raw), 4, 38)
            
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
    dataset_path = r"d:\eeg-minirocket-project\physionet"
    models_dir = r"d:\eeg-minirocket-project\eeg-mi-bci\models"
    os.makedirs(models_dir, exist_ok=True)
    
    print(json.dumps({"type": "info", "message": "Starting Incremental Training across 109 subjects..."}), flush=True)
    
    # Initialize Models
    mr_pipeline = MiniRocketPipeline()
    cnn_pipeline = CNN_LSTM_Pipeline(channels=20, num_classes=4, epochs=5, lr=3e-3, batch_size=16)
    
    minirocket_initialized = False
    
    mr_total_acc = 0.0
    cnn_total_acc = 0.0
    valid_subs = 0
    
    for sub in range(1, 110):
        if sub in [88, 89, 92, 100, 104, 106]:
            continue
            
        print(json.dumps({"type": "progress", "message": f"Extracting Subject {sub} / 109..."}), flush=True)
        X, y = extract_subject_data(sub, dataset_path)
        
        if X is None or len(X) == 0:
            continue
        
        # Train MiniRocket (Incremental)
        if not minirocket_initialized:
            print(json.dumps({"type": "progress", "message": f"Initializing MiniRocket kernels on Sub {sub}..."}), flush=True)
            mr_pipeline.fit(X, y)
            minirocket_initialized = True
            
            from sklearn.metrics import accuracy_score
            preds = mr_pipeline.predict(X)
            acc = accuracy_score(y, preds)
            mr_total_acc += acc
        else:
            print(json.dumps({"type": "progress", "message": f"Partial fitting MiniRocket on Sub {sub}..."}), flush=True)
            X_3d = X[:, np.newaxis, :] if X.ndim == 2 else X
            X_features = mr_pipeline.transform.transform(X_3d)
            # Crucial missing step: Update the scaler with the new data distribution
            mr_pipeline.scaler.partial_fit(X_features)
            X_features = mr_pipeline.scaler.transform(X_features)
            mr_pipeline.classifier.partial_fit(X_features, y, classes=np.array([0, 1, 2, 3]))
            preds = mr_pipeline.classifier.predict(X_features)
            
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y, preds)
            mr_total_acc += acc
            
        # Train CNN-LSTM (Sequential epochs)
        print(json.dumps({"type": "progress", "message": f"Fitting CNN on Sub {sub}..."}), flush=True)
        def cnn_progress(ep, t_loss, t_acc, v_loss, v_acc):
            pass 
            
        # Pass incremental=True to accumulate normalisation mean/std
        cnn_pipeline.fit(X, y, X, y, progress_callback=cnn_progress, incremental=True)
        
        preds_cnn = cnn_pipeline.predict(X)
        acc_cnn = accuracy_score(y, np.clip(preds_cnn, 0, 3))
        cnn_total_acc += acc_cnn
        
        valid_subs += 1
        
        # Every 5 subjects, save a checkpoint model
        if valid_subs % 5 == 0 or sub == 109:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            mr_path = os.path.join(models_dir, f"master_incremental_minirocket_subs{valid_subs}_{timestamp}.pkl")
            cnn_path = os.path.join(models_dir, f"master_incremental_cnn_subs{valid_subs}_{timestamp}.pth")
            mr_pipeline.save(mr_path)
            cnn_pipeline.save(cnn_path)
            
            print(json.dumps({"type": "info", "message": f"Checkpoint saved at Subject {sub}. Avg MiniRocket Acc: {mr_total_acc/valid_subs:.2f}, CNN Acc: {cnn_total_acc/valid_subs:.2f}"}), flush=True)
            
    print(json.dumps({"type": "complete", "message": "Incremental training across all subjects finished successfully!"}), flush=True)
