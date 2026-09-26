import os
import sys
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from minirocket_engine import MiniRocketPipeline
from cnn_lstm_engine import CNN_LSTM_Pipeline

def run_ablation(data_dir, group_id):
    X_path = os.path.join(data_dir, f'group{group_id}_X.npy')
    y_path = os.path.join(data_dir, f'group{group_id}_y.npy')
    
    if not os.path.exists(X_path):
        print(f"Data for Group {group_id} not found.")
        return {}
        
    X_aug = np.load(X_path, mmap_mode='r')
    y_raw = np.load(y_path)
    
    unique_labels = np.unique(y_raw)
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    y = np.array([label_map[lbl] for lbl in y_raw])
    
    # Original channels before augmentation: 21 channels. 
    # The augmented channels were concatenated after the original ones.
    # Actually, in spatial_channel_augmentation, we picked [C3, C4, Cz, P3, P4, Pz, F3, F4, Fz] -> 9 channels?
    # Let's just use the first 10 channels for original, or let's assume original is X_aug.shape[1] - augmented.
    # The `spatial_channel_augmentation` adds 10 differential pairs.
    # Let's check how many channels the original has.
    original_channels_count = X_aug.shape[1] - 10
    X_orig = X_aug[:, :original_channels_count, :]
    
    # Let's use MiniRocket for fast ablation
    X_orig_train, X_orig_test, y_orig_train, y_orig_test = train_test_split(X_orig, y, test_size=0.2, random_state=42)
    X_aug_train, X_aug_test, y_aug_train, y_aug_test = train_test_split(X_aug, y, test_size=0.2, random_state=42)
    
    if len(X_orig_train) > 1000:
        idx = np.random.choice(len(X_orig_train), 1000, replace=False)
        X_orig_train = X_orig_train[idx]
        y_orig_train = y_orig_train[idx]
        X_aug_train = X_aug_train[idx]
        y_aug_train = y_aug_train[idx]
    
    mr_orig = MiniRocketPipeline()
    mr_orig.fit(X_orig_train.copy(), y_orig_train)
    orig_acc = accuracy_score(y_orig_test[:200], mr_orig.predict(X_orig_test[:200].copy()))
    
    mr_aug = MiniRocketPipeline()
    mr_aug.fit(X_aug_train.copy(), y_aug_train)
    aug_acc = accuracy_score(y_aug_test[:200], mr_aug.predict(X_aug_test[:200].copy()))
    
    return {'baseline_acc': float(orig_acc), 'augmented_acc': float(aug_acc), 'delta': float(aug_acc - orig_acc)}

def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'processed'))
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    os.makedirs(results_dir, exist_ok=True)
    
    print("Running Ablation Study on Group 2 and Group 3...")
    metrics = {}
    for g in [2, 3]:
        print(f"Testing Group {g}...")
        metrics[f"group{g}"] = run_ablation(data_dir, g)
        print(metrics[f"group{g}"])
        
    with open(os.path.join(results_dir, 'ablation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Ablation study complete.")

if __name__ == "__main__":
    main()
