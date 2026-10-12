import sys
sys.path.insert(0, r"D:\pip_packages")
import os
import json
import glob
import numpy as np
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Ensure src is in path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from minirocket_engine import MiniRocketPipeline
from cnn_lstm_engine import CNN_LSTM_Pipeline
from advanced_eeg_engine import AdvancedEEGPipeline
from eegnet_engine import EEGNet_Pipeline
from convnets_engine import ConvNet_Pipeline

def get_test_split(X, y, meta):
    X_test, y_test = [], []
    if meta is not None and meta.ndim > 0 and len(meta) == len(y):
        for subject_id in np.unique([m['subject_id'] for m in meta]):
            sub_mask = np.array([m['subject_id'] == subject_id for m in meta])
            sub_X = X[sub_mask]
            sub_y = y[sub_mask]
            sub_meta = meta[sub_mask]
            
            sub_runs = np.unique([m['run_id'] for m in sub_meta])
            sub_runs = np.sort(sub_runs)
            
            if len(sub_runs) >= 3:
                test_runs = [sub_runs[-1]]
            elif len(sub_runs) == 2:
                test_runs = [sub_runs[-1]]
            else:
                test_runs = []
                
            for i, m in enumerate(sub_meta):
                if m['run_id'] in test_runs:
                    X_test.append(sub_X[i])
                    y_test.append(sub_y[i])
        
        X_test = np.array(X_test) if X_test else np.empty((0, *X.shape[1:]))
        y_test = np.array(y_test) if y_test else np.empty((0,))
    else:
        # No metadata, use temporal split 90-10
        if y.ndim > 1 or np.issubdtype(y.dtype, np.floating):
            test_split = int(len(y) * 0.9)
            X_test, y_test = X[test_split:], y[test_split:]
        else:
            X_test_list, y_test_list = [], []
            for cls in np.unique(y):
                idx = np.where(y == cls)[0]
                test_split = int(len(idx) * 0.9)
                X_test_list.append(X[idx[test_split:]])
                y_test_list.append(y[idx[test_split:]])
            X_test = np.concatenate(X_test_list, axis=0) if X_test_list else np.empty((0, *X.shape[1:]))
            y_test = np.concatenate(y_test_list, axis=0) if y_test_list else np.empty((0,))
            
    return X_test, y_test

def evaluate_models():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    data_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    
    if not os.path.exists(models_dir):
        print("No models directory found.")
        return
        
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    if not model_files:
        print("No .pth models found in models/ directory.")
        return
        
    print(f"{'='*100}")
    print(f"EVALUATION OF ALL MODELS ACROSS ALL DATASETS")
    print(f"{'='*100}")
    print(f"{'Model File':<60} | {'Dataset':<15} | {'Test Size':<10} | {'Accuracy':<10}")
    print(f"{'-'*100}")
    
    # Group by dataset
    data_files = glob.glob(os.path.join(data_dir, "*.npz"))
    
    dataset_to_datafiles = {}
    for df in data_files:
        name = os.path.basename(df).lower()
        key = None
        if "physionet" in name: key = "physionet"
        elif "bnci" in name or "bci" in name or "2a" in name: key = "bci2a"
        elif "highgamma" in name: key = "highgamma"
        elif "kaya" in name: key = "kaya"
        elif "way" in name: key = "way"
        if key:
            if key not in dataset_to_datafiles: dataset_to_datafiles[key] = []
            dataset_to_datafiles[key].append(df)

    # Cache loaded data to prevent reloading
    loaded_data = {}

    for mf in sorted(model_files):
        basename = os.path.basename(mf)
        parts = basename.replace(".pth", "").split("_")
        if len(parts) < 3: continue
        
        # Parse model file name master_{dataset}_{arch}_...
        prefix = parts[0]
        ds_hint = parts[1]
        
        # Arch string could have underscores e.g. cnn_lstm, gpu_minirocket
        arch_hint = None
        if "minirocket" in basename: arch_hint = "minirocket"
        elif "cnn_lstm" in basename: arch_hint = "cnn_lstm"
        elif "conformer" in basename: arch_hint = "conformer"
        elif "eegnet" in basename: arch_hint = "eegnet"
        elif "shallow" in basename: arch_hint = "shallow"
        
        if not arch_hint or ds_hint not in dataset_to_datafiles:
            continue
            
        data_paths = dataset_to_datafiles[ds_hint]
        
        success = False
        for data_path in data_paths:
            if data_path not in loaded_data:
                f32_path = data_path.replace('.npz', '_f32.npz')
                if os.path.exists(f32_path):
                    loaded = np.load(f32_path, allow_pickle=True)
                    X = loaded['X']
                    y = np.array(loaded['y'])
                else:
                    loaded = np.load(data_path, mmap_mode='r', allow_pickle=True)
                    X = np.array(loaded['X'], dtype=np.float32)
                    y = np.array(loaded['y'])
                    
                meta = loaded.get('meta')
                X_test, y_test = get_test_split(X, y, meta)
                loaded_data[data_path] = (X_test, y_test)
                
            X_test, y_test = loaded_data[data_path]
            
            if len(X_test) == 0:
                continue
                
            # Instantiate model
            pipeline = None
            is_regression = (y_test.ndim > 1 or np.issubdtype(y_test.dtype, np.floating))
            task_type = "regression" if is_regression else "classification"
            num_cls = y_test.shape[1] if is_regression else len(np.unique(y_test))
            
            try:
                if arch_hint == "minirocket":
                    pipeline = MiniRocketPipeline()
                elif arch_hint == "cnn_lstm":
                    pipeline = CNN_LSTM_Pipeline(num_classes=num_cls, channels=X_test.shape[1], samples=X_test.shape[2], task_type=task_type)
                elif arch_hint == "conformer":
                    pipeline = AdvancedEEGPipeline(num_classes=num_cls, channels=X_test.shape[1], samples=X_test.shape[2], task_type=task_type)
                elif arch_hint == "eegnet":
                    pipeline = EEGNet_Pipeline(num_classes=num_cls, channels=X_test.shape[1], samples=X_test.shape[2], task_type=task_type)
                elif arch_hint == "shallow":
                    pipeline = ConvNet_Pipeline(arch="shallow", num_classes=num_cls, channels=X_test.shape[1], samples=X_test.shape[2]) # task_type removed
                    
                pipeline.load(mf)
                
                y_pred = pipeline.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                acc_str = f"{acc*100:.2f}%"
                
                print(f"{basename:<60} | {ds_hint:<15} | {len(y_test):<10} | {acc_str:<10}")
                
                # Print classes
                unique_preds = np.unique(y_pred)
                unique_true = np.unique(y_test)
                if len(unique_preds) < len(unique_true):
                    print(f"  -> WARNING: Model predicted {len(unique_preds)} classes {unique_preds}, but {len(unique_true)} were expected {unique_true}")
                
                success = True
                break # Found correct data file that doesn't throw shape error
            except Exception as e:
                # Keep trying other datasets
                err_msg = str(e)
                continue
                
        if not success:
            print(f"{basename:<60} | {ds_hint:<15} | ERROR: Incompatible shapes or {err_msg[:30]}")
            
    print(f"{'='*100}")

if __name__ == '__main__':
    evaluate_models()
