import os
import sys
import glob
import numpy as np
import datetime

sys.path.insert(0, os.path.abspath('src'))
from minirocket_engine import MiniRocketPipeline

def get_test_split(X, y, meta):
    if meta is not None and meta.ndim > 0 and len(meta) == len(y):
        X_train, y_train, X_test, y_test = [], [], [], []
        for subject_id in np.unique([m['subject_id'] for m in meta]):
            sub_mask = np.array([m['subject_id'] == subject_id for m in meta])
            sub_X = X[sub_mask]
            sub_y = y[sub_mask]
            sub_meta = meta[sub_mask]
            sub_runs = np.unique([m['run_id'] for m in sub_meta])
            sub_runs = np.sort(sub_runs)
            
            if len(sub_runs) >= 3:
                test_runs = [sub_runs[-1]]
                train_runs = sub_runs[:-1]
            elif len(sub_runs) == 2:
                test_runs = [sub_runs[-1]]
                train_runs = [sub_runs[0]]
            else:
                train_runs = sub_runs
                test_runs = []
                
            for i, m in enumerate(sub_meta):
                if m['run_id'] in test_runs:
                    X_test.append(sub_X[i])
                    y_test.append(sub_y[i])
                else:
                    X_train.append(sub_X[i])
                    y_train.append(sub_y[i])
        return (np.array(X_train) if X_train else np.empty((0, *X.shape[1:]))), \
               (np.array(y_train) if y_train else np.empty((0,))), \
               (np.array(X_test) if X_test else np.empty((0, *X.shape[1:]))), \
               (np.array(y_test) if y_test else np.empty((0,)))
    else:
        # Fallback chronological split
        X_train_list, y_train_list, X_test_list, y_test_list = [], [], [], []
        for cls in np.unique(y):
            idx = np.where(y == cls)[0]
            split = int(len(idx) * 0.9)
            X_train_list.append(X[idx[:split]])
            y_train_list.append(y[idx[:split]])
            X_test_list.append(X[idx[split:]])
            y_test_list.append(y[idx[split:]])
        return np.concatenate(X_train_list, axis=0), np.concatenate(y_train_list, axis=0), \
               np.concatenate(X_test_list, axis=0), np.concatenate(y_test_list, axis=0)

data_files = glob.glob(os.path.join('data', 'processed', '*.npz'))
# Filter to get best files (sub1to10 preferred)
best_files = {}
for df in data_files:
    if "f32" in df: continue
    name = os.path.basename(df).lower()
    key = None
    if "physionet" in name: key = "physionet"
    elif "bnci" in name or "bci" in name or "2a" in name: key = "bci2a"
    elif "highgamma" in name: key = "highgamma"
    elif "kaya" in name: key = "kaya"
    elif "way" in name: key = "way"
    elif "dreamer" in name: key = "dreamer"
    
    if key:
        # Prefer 1to10 over 1to1
        if key not in best_files:
            best_files[key] = df
        else:
            if "1to10" in name and "1to10" not in best_files[key]:
                best_files[key] = df
            elif "10to10" not in name and "10to10" in best_files[key]:
                best_files[key] = df

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
models_dir = 'models'

for ds_hint, data_path in best_files.items():
    print(f"Training MiniRocket on {ds_hint} using {data_path}...")
    loaded = np.load(data_path, allow_pickle=True)
    X = np.array(loaded['X'], dtype=np.float32)
    y = np.array(loaded['y'])
    meta = loaded.get('meta')
    
    X_train, y_train, X_test, y_test = get_test_split(X, y, meta)
    
    if len(X_train) == 0:
        continue
        
    print(f"  Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    mr_pipeline = MiniRocketPipeline(num_kernels=10000, in_channels=X_train.shape[1], seq_len=X_train.shape[2], head_epochs=40, head_lr=1e-3)
    mr_pipeline.sfreq = 160.0
    mr_pipeline.channel_names = [f"EEG_{i}" for i in range(X_train.shape[1])]
    
    mr_pipeline.fit(X_train, y_train)
    
    # Save model
    model_name = f"master_{ds_hint}_gpu_minirocket_subs1to10_{timestamp}.pth"
    mr_pipeline.save(os.path.join(models_dir, model_name))
    
    # Evaluate
    y_pred = mr_pipeline.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"  --> Final Test Accuracy: {acc*100:.2f}%\n")
    
print("All done!")
