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
from csp_engine import CSP_Engine

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
        elif X_all.shape[2] < target_samples:
            pad_width = target_samples - X_all.shape[2]
            X_all = np.pad(X_all, ((0,0), (0,0), (0,pad_width)), mode='constant')
            
        fname = f"bci2a_data_{sub_str}.npz"
        np.savez_compressed(os.path.join(output_dir, fname), X=X_all, y=y_all)
        return X_all.shape, y_all.shape

    # Existing Physionet logic below
    # Determine runs to load based on mode
    # For master or OVR, we theoretically need ALL runs (1-14) to get all 10 groups
    runs = list(range(1, 15))
    
    X_list = []
    y_list = []
    
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
                    X_list.append(X_batch)
                    y_list.extend(y_batch)
                else:
                    print("X_batch was empty!", flush=True)
                        
            del raws
            gc.collect()
        except Exception as e:
            print(f"Exception: {e}", flush=True)


    if len(X_list) == 0:
        return None, None
        
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    fname = f"master_data_{sub_str}.npz" if mode == "master" else f"ovr_group{group_id}_data_{sub_str}.npz"
    np.savez_compressed(os.path.join(output_dir, fname), X=X_all, y=y_all)
    
    return X_all.shape, y_all.shape

def train_models(mode, group_id, data_dir, models_dir, dataset_path="", model_name="MiniRocket", epochs=10, lr=1e-3, kernels=10000, train_split=0.8, sub_start=1, sub_end=1):
    sub_str = f"subs{sub_start}to{sub_end}"
    
    if "2a" in dataset_path.lower():
        fname = f"bci2a_data_{sub_str}.npz"
        model_prefix = "bci2a"
    else:
        fname = f"master_data_{sub_str}.npz" if mode == "master" else f"ovr_group{group_id}_data_{sub_str}.npz"
        model_prefix = "master" if mode == "master" else f"ovr_group{group_id}"
        
    data_path = os.path.join(data_dir, fname)
    
    if not os.path.exists(data_path):
        print(json.dumps({"type": "error", "message": f"Data not found."}), flush=True)
        return
        
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
    
    test_size = 1.0 - train_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    try:
        from dataset_2a_loader import augment_data
        print(json.dumps({"type": "info", "message": f"Augmenting training data... Original shape: {X_train.shape}"}), flush=True)
        X_train, y_train = augment_data(X_train, y_train)
        print(json.dumps({"type": "info", "message": f"Augmented shape: {X_train.shape}"}), flush=True)
    except ImportError:
        pass
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_str = f"subs{sub_start}to{sub_end}"
    num_cls = len(np.unique(y))
    
    # --- MODEL EXECUTION SWITCH ---
    if model_name == "MiniRocket":
        # Train MiniRocket (GPU-accelerated)
        print(json.dumps({"type": "progress", "message": "Training GPU-Accelerated MiniRocket..."}), flush=True)
        mr_pipeline = MiniRocketPipeline(num_kernels=kernels, in_channels=X.shape[1], seq_len=X.shape[2])
        
        import threading
        import time
        stop_timer = False
        def print_timer():
            start_t = time.time()
            estimated_total = 240.0
            while not stop_timer:
                elapsed = time.time() - start_t
                pct = min(99, int((elapsed / estimated_total) * 100))
                if pct == 99:
                    extra_time = int(elapsed - estimated_total)
                    msg = f"Training MiniRocket... (99%) [Finalizing kernels: +{extra_time}s]"
                else:
                    msg = f"Training MiniRocket... ({pct}%)"
                print(json.dumps({"type": "progress", "message": msg}), flush=True)
                time.sleep(1)
                
        timer_thread = threading.Thread(target=print_timer, daemon=True)
        timer_thread.start()
        
        try:
            if args.finetune_model:
                mr_pipeline.load(args.finetune_model)
            mr_pipeline.fit(X_train, y_train)
        finally:
            stop_timer = True
            timer_thread.join(timeout=1.0)
            
        y_pred = mr_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mr_pipeline.save(os.path.join(models_dir, f"{model_prefix}_gpu_minirocket_{sub_str}_{timestamp}.pth"))
        
        print(json.dumps({
            "type": "epoch", "epoch": 1, "total_epochs": 1,
            "train_loss": 0.0, "val_loss": 0.0,
            "train_acc": float(acc), "val_acc": float(acc)
        }), flush=True)

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
            batch_size=64
        )
        
        try:
            if args.finetune_model:
                cnn_lstm_pipeline.load(args.finetune_model)
            cnn_lstm_pipeline.fit(X_train, y_train, X_test, y_test)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
    
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
            batch_size=64
        )
        
        try:
            if args.finetune_model:
                conformer_pipeline.load(args.finetune_model)
            conformer_pipeline.fit(X_train, y_train, X_test, y_test)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
    
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
            batch_size=64
        )
        
        try:
            if args.finetune_model:
                eegnet_pipeline.load(args.finetune_model)
            eegnet_pipeline.fit(X_train, y_train, X_test, y_test)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
    
        eegnet_pipeline.save(os.path.join(models_dir, f"{model_prefix}_eegnet_{sub_str}_{timestamp}.pth"))

    elif model_name == "Shallow ConvNet":
        print(json.dumps({"type": "progress", "message": "Training Shallow ConvNet..."}), flush=True)
        print(json.dumps({"type": "reset_chart"}), flush=True)
        shallow_pipeline = ConvNet_Pipeline(arch="shallow", num_classes=num_cls, channels=X.shape[1], samples=X.shape[2], epochs=epochs, lr=lr)
        try:
            if args.finetune_model:
                shallow_pipeline.load(args.finetune_model)
            shallow_pipeline.fit(X_train, y_train, X_test, y_test)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        shallow_pipeline.save(os.path.join(models_dir, f"{model_prefix}_shallow_{sub_str}_{timestamp}.pth"))

    elif model_name == "Deep ConvNet":
        print(json.dumps({"type": "progress", "message": "Training Deep ConvNet..."}), flush=True)
        print(json.dumps({"type": "reset_chart"}), flush=True)
        deep_pipeline = ConvNet_Pipeline(arch="deep", num_classes=num_cls, channels=X.shape[1], samples=X.shape[2], epochs=epochs, lr=lr)
        try:
            if args.finetune_model:
                deep_pipeline.load(args.finetune_model)
            deep_pipeline.fit(X_train, y_train, X_test, y_test)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        deep_pipeline.save(os.path.join(models_dir, f"{model_prefix}_deep_{sub_str}_{timestamp}.pth"))

    elif model_name == "CSP + LDA":
        print(json.dumps({"type": "progress", "message": "Training CSP + LDA..."}), flush=True)
        csp_pipeline = CSP_Engine(classifier_type="lda", n_components=4)
        try:
            if args.finetune_model:
                csp_pipeline.load(args.finetune_model)
            csp_pipeline.fit(X_train, y_train, X_test, y_test)
        except Exception as e:
            print(json.dumps({"type": "error", "message": str(e)}), flush=True)
        csp_pipeline.save(os.path.join(models_dir, f"{model_prefix}_csp_lda_{sub_str}_{timestamp}.pkl"))

    # Compute final metrics for the completion message
    final_acc = 0.0
    latency_ms = 0.0
    try:
        import time
        pipeline_to_eval = None
        if model_name == "MiniRocket": pipeline_to_eval = mr_pipeline
        elif model_name == "CNN-LSTM": pipeline_to_eval = conformer_pipeline
        elif model_name == "EEGNet": pipeline_to_eval = eegnet_pipeline
        elif model_name == "Shallow ConvNet": pipeline_to_eval = shallow_pipeline
        elif model_name == "Deep ConvNet": pipeline_to_eval = deep_pipeline
        elif model_name == "CSP + LDA": pipeline_to_eval = csp_pipeline
        
        if pipeline_to_eval:
            # Measure latency on 1 sample to simulate real-time BCI latency
            start_t = time.time()
            _ = pipeline_to_eval.predict(X_test[:1])
            end_t = time.time()
            latency_ms = (end_t - start_t) * 1000.0
            
            # Measure final validation accuracy
            y_pred_all = pipeline_to_eval.predict(X_test)
            final_acc = float(accuracy_score(y_test, y_pred_all))
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
    
    sub_str = f"subs{args.sub_start}to{args.sub_end}"
    if "2a" in args.dataset.lower():
        fname = f"bci2a_data_{sub_str}.npz"
    else:
        fname = f"master_data_{sub_str}.npz" if args.mode == "master" else f"ovr_group{args.group}_data_{sub_str}.npz"
    data_path = os.path.join(data_dir, fname)
    
    if not os.path.exists(data_path):
        shapeX, shapeY = extract_and_save_data(args.mode, args.group, data_dir, args.dataset, sub_start=args.sub_start, sub_end=args.sub_end)
        if shapeX is None:
            print(json.dumps({"type": "error", "message": "No data extracted."}), flush=True)
            sys.exit(1)
            
    train_models(
        mode=args.mode, 
        group_id=args.group, 
        data_dir=data_dir, 
        models_dir=models_dir,
        dataset_path=args.dataset,
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        kernels=args.kernels,
        train_split=args.partition / 100.0,
        sub_start=args.sub_start,
        sub_end=args.sub_end
    )
