import os
import sys
import numpy as np
import json
import gc
import shutil
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from binary_parser import load_local_eeg_data
from preprocessing import apply_car, apply_bandpass_filter, spatial_channel_augmentation, epoch_and_segment
from minirocket_engine import MiniRocketPipeline
from cnn_lstm_engine import CNN_LSTM_Pipeline

def archive_old_models(models_dir):
    """
    Moves existing .pkl and .pth models to an archive folder with a timestamp.
    """
    files_to_archive = [f for f in os.listdir(models_dir) if f.endswith('.pkl') or f.endswith('.pth')]
    if files_to_archive:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_dir = os.path.join(models_dir, 'archive', f'session_{timestamp}')
        os.makedirs(archive_dir, exist_ok=True)
        for f in files_to_archive:
            shutil.move(os.path.join(models_dir, f), os.path.join(archive_dir, f))
        print(f"Archived {len(files_to_archive)} old models to {archive_dir}")

def extract_and_save_group_data(group_id, output_dir, subject_range=range(1, 110), dataset_path=r'd:\eeg-minirocket-project\physionet'):
    """
    Extracts all trials for a given group, saving them iteratively to avoid RAM issues.
    """
    group_runs = {
        1: [1, 2],
        2: [3, 7, 11],
        3: [4, 8, 12],
        4: [5, 9, 13],
        5: [6, 10, 14]
    }
    
    runs = group_runs[group_id]
    
    X_list = []
    y_list = []
    
    print(json.dumps({"type": "progress", "model": "Extraction", "message": f"Extracting data for Group {group_id}..."}), flush=True)
    for sub in subject_range:
        if sub in [88, 89, 92, 100, 104, 106]:
            continue
        raws, events_list, mappings = load_local_eeg_data(sub, runs, data_dir=dataset_path)
        for i, raw in enumerate(raws):
            events = events_list[i]
            mapping = mappings[i]
            
            raw = apply_bandpass_filter(apply_car(raw), 4, 38)
            raw = spatial_channel_augmentation(raw)
            
            event_id = {v: k for k, v in mapping.items()}
            
            # Epoch and segment with group_id to preserve T0 for group 1
            X_batch, y_batch = epoch_and_segment(raw, events, event_id, tmin=0, tmax=4.1, group_id=group_id)
                
            if X_batch.shape[0] > 0:
                X_list.append(X_batch)
                y_list.append(y_batch)
            
        del raws
        gc.collect()

    if len(X_list) == 0:
        return None, None
        
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    
    os.makedirs(output_dir, exist_ok=True)
    # Save as compressed .npz archive
    np.savez_compressed(os.path.join(output_dir, f'group{group_id}_data.npz'), X=X_all, y=y_all)
    
    return X_all.shape, y_all.shape

def train_group_models(group_id, data_dir, models_dir):
    data_path = os.path.join(data_dir, f'group{group_id}_data.npz')
    
    if not os.path.exists(data_path):
        print(json.dumps({"type": "error", "message": f"Data for Group {group_id} not found."}), flush=True)
        return {}
        
    loaded = np.load(data_path)
    X = loaded['X']
    y_raw = loaded['y']
    
    # Remap y to 0-indexed integers
    unique_labels = np.unique(y_raw)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in y_raw])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MiniRocket
    print(json.dumps({"type": "progress", "model": "MiniRocket", "message": f"Training MiniRocket for Group {group_id}..."}), flush=True)
    minirocket = MiniRocketPipeline()
    if len(X_train) > 1000:
        idx = np.random.choice(len(X_train), 1000, replace=False)
        mr_X = X_train[idx].copy()
        mr_y = y_train[idx]
    else:
        mr_X = X_train.copy()
        mr_y = y_train
        
    minirocket.fit(mr_X, mr_y)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    mr_name = f'group{group_id}_minirocket_rest_removed_{date_str}.pkl'
    minirocket.save(os.path.join(models_dir, mr_name))
    
    mr_preds = minirocket.predict(X_test[:200].copy())
    mr_acc = accuracy_score(y_test[:200], mr_preds)
    
    print(json.dumps({"type": "metric", "model": "MiniRocket", "val_acc": float(mr_acc)}), flush=True)
    
    # CNN
    print(json.dumps({"type": "progress", "model": "CNN-LSTM", "message": f"Training CNN for Group {group_id}..."}), flush=True)
    cnn = CNN_LSTM_Pipeline(channels=X_train.shape[1], num_classes=len(unique_labels), epochs=10, batch_size=32)
    
    def progress_callback(epoch, train_loss, train_acc, val_loss, val_acc):
        print(json.dumps({
            "type": "epoch",
            "model": "CNN-LSTM",
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        }), flush=True)
        
    cnn.fit(X_train.copy(), y_train, progress_callback=progress_callback)
    cnn_name = f'group{group_id}_cnn_rest_removed_{date_str}.pth'
    cnn.save(os.path.join(models_dir, cnn_name))
    
    cnn_preds = cnn.predict(X_test[:200].copy())
    cnn_acc = accuracy_score(y_test[:200], cnn_preds)
    
    return {'minirocket_acc': float(mr_acc), 'cnn_acc': float(cnn_acc)}

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=int, help='Group ID to train (1-5)')
    parser.add_argument('--dataset', type=str, default=r'd:\eeg-minirocket-project\physionet', help='Path to the PhysioNet dataset')
    args = parser.parse_args()

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    
    # Train on the full 109 subjects
    subject_range = range(1, 110) 
    
    if args.group:
        g = args.group
        print(json.dumps({"type": "info", "message": f"Starting single-group training for Group {g}"}), flush=True)
        shape_X, shape_y = extract_and_save_group_data(g, data_dir, subject_range=subject_range, dataset_path=args.dataset)
        if shape_X is not None:
            print(json.dumps({"type": "info", "message": f"Extracted shape: {shape_X}"}), flush=True)
            train_group_models(g, data_dir, models_dir)
        print(json.dumps({"type": "complete", "message": f"Group {g} training complete."}), flush=True)
    else:
        # Default loop behaviour
        # Archive old models first
        archive_old_models(models_dir)
        metrics = {}
        for g in range(1, 6):
            print(f"--- Group {g} ---")
            shape_X, shape_y = extract_and_save_group_data(g, data_dir, subject_range=subject_range)
            if shape_X is None:
                continue
            m = train_group_models(g, data_dir, models_dir)
            metrics[f"group{g}"] = m
            
        with open(os.path.join(models_dir, '..', 'results', 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)
            
        print("Training complete.")

if __name__ == "__main__":
    main()
