import os
import glob
import numpy as np
import mne

def load_high_gamma_data(data_dir, subject_range, resample_freq=160.0):
    print(f"Loading REAL HIGH-GAMMA data for subjects {subject_range}...")
    
    all_X = []
    all_y = []
    
    # 5 classes (Rest, Hand Up, Hand Down, Hand Left, Hand Right)
    for ext in ['*.edf', '**/*.edf']:
        files = glob.glob(os.path.join(data_dir, ext), recursive=True)
        for f in files:
            try:
                raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
                if resample_freq:
                    raw.resample(resample_freq)
                
                # Extract events (assuming standard annotations exist in the EDF)
                events, event_dict = mne.events_from_annotations(raw, verbose=False)
                epochs = mne.Epochs(raw, events, tmin=0, tmax=4.0, baseline=None, preload=True, verbose=False)
                
                X = epochs.get_data() # (trials, channels, time)
                y = epochs.events[:, 2] # Event IDs
                
                all_X.append(X)
                all_y.append(y)
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")
                
    if not all_X:
        return np.array([]), np.array([])
        
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    
    # Ensure targets are 0-indexed for PyTorch (0 to 4)
    unique_y = np.unique(y_all)
    y_mapped = np.zeros_like(y_all)
    for i, val in enumerate(unique_y[:5]):
        y_mapped[y_all == val] = i
        
    return X_all, y_mapped
