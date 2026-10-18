import os
import glob
import numpy as np
import pandas as pd

def load_way_data(data_dir, subject_range, resample_freq=160.0):
    print(f"Loading REAL WAY-EEG-GAL data for subjects {subject_range}...")
    
    all_X = []
    all_y = []
    
    # Example logic for grasping data (CSVs)
    for ext in ['*.csv', '**/*.csv']:
        files = glob.glob(os.path.join(data_dir, ext), recursive=True)
        # Sort files to match data with events if needed, assuming simple structure for now
        for f in files:
            try:
                if 'events' in f.lower():
                    continue # Skip event files in the main loop, handle them paired
                    
                # Find matching event file
                event_file = f.replace('data', 'events')
                if not os.path.exists(event_file):
                    continue
                    
                df_data = pd.read_csv(f)
                df_events = pd.read_csv(event_file)
                
                # Exclude id column
                eeg_data = df_data.drop('id', axis=1, errors='ignore').values
                event_data = df_events.drop('id', axis=1, errors='ignore').values
                
                # Convert continuous event data to epochs (simplified sliding window)
                window_size = int(resample_freq * 4) # 4 seconds
                stride = window_size // 2
                
                for start in range(0, len(eeg_data) - window_size, stride):
                    end = start + window_size
                    window_X = eeg_data[start:end].T # (channels, time)
                    
                    # Get the most common event in this window
                    window_y = event_data[start:end]
                    # WAY events are one-hot encoded in 6 columns.
                    # Find if any event is active (sum > 0), else label 0 (Rest)
                    active_events = np.sum(window_y, axis=0)
                    if np.max(active_events) > (window_size * 0.1): # 10% active
                        label = np.argmax(active_events) + 1 # 1 to 6
                    else:
                        label = 0
                        
                    all_X.append(window_X)
                    all_y.append(label)
                    
            except Exception as e:
                print(f"Skipping {f} due to error: {e}")
                
    if not all_X:
        return np.array([]), np.array([])
        
    X_all = np.array(all_X) # (trials, channels, time)
    y_all = np.array(all_y)
    
    return X_all, y_all
