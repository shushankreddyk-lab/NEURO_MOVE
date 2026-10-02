import numpy as np
import mne
import os
from moabb.datasets import BNCI2014_001, PhysionetMI, Schirrmeister2017
from moabb.paradigms import MotorImagery

def load_dataset(dataset_name, data_dir=r"D:\eeg-minirocket-project\data", subject_id=1):
    """
    Unified loader for all 5 EEG datasets.
    Returns:
        X_train, y_train, X_test, y_test
        Shapes: X -> (trials, channels, timepoints), y -> (trials,)
    """
    print(f"Loading Dataset: {dataset_name} for Subject {subject_id}...")
    
    # Configuration for MOABB to use the correct data directory
    mne.set_config('MNE_DATASETS_MOABB_PATH', data_dir)
    
    # -----------------------------------------------------------------
    # 1. BCI Competition IV 2a (Tongue, Left Hand, Right Hand, Feet)
    # -----------------------------------------------------------------
    if dataset_name == "BNCI2014_001":
        dataset = BNCI2014_001()
        dataset.subject_list = [subject_id]
        paradigm = MotorImagery(n_classes=4)
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
        
        mask = (y != 'rest')
        X, y = X[mask], y[mask]
        
        # Simple split (first 70% train, last 30% test)
        split_idx = int(len(X) * 0.7)
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
        
    # -----------------------------------------------------------------
    # 2. PhysioNet MI (Up, Down, Left, Right - Fists/Feet)
    # -----------------------------------------------------------------
    elif dataset_name == "PhysionetMI":
        dataset = PhysionetMI()
        dataset.subject_list = [subject_id]
        paradigm = MotorImagery(n_classes=4)
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
        
        # Remove 'rest' class as requested by user
        mask = (y != 'rest')
        X, y = X[mask], y[mask]
        
        split_idx = int(len(X) * 0.7)
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]

    # -----------------------------------------------------------------
    # 3. High-Gamma / Schirrmeister 2017 (Spread Fingers vs Fist)
    # -----------------------------------------------------------------
    elif dataset_name == "HighGamma":
        dataset = Schirrmeister2017()
        dataset.subject_list = [subject_id]
        paradigm = MotorImagery(n_classes=4)
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
        
        mask = (y != 'rest')
        X, y = X[mask], y[mask]
        
        split_idx = int(len(X) * 0.7)
        return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]
        
    # -----------------------------------------------------------------
    # 4. Kaya Finger Movements (Thumb, Index, Middle, etc.)
    # -----------------------------------------------------------------
    elif dataset_name == "KayaFingers":
        import scipy.io
        import glob
        kaya_dir = os.path.join(data_dir, "Kaya_Finger_Movements")
        print(f"Loading Kaya dataset from {kaya_dir}...")
        
        # In full production we loop through all files. Just loading one for demo.
        # Ensure we pick a fully downloaded, uncorrupted file (e.g. 5F-SubjectB)
        mat_files = [f for f in glob.glob(os.path.join(kaya_dir, "*.mat")) if "5F-" in f]
        if not mat_files:
            raise FileNotFoundError(f"No valid 5F .mat files found in {kaya_dir}")
            
        mat_data = scipy.io.loadmat(mat_files[0])
        
        # Extract data from the 'o' struct
        o = mat_data['o'][0, 0]
        data = o['data']      # Shape: (Timepoints, Channels)
        marker = o['marker'].flatten() # Shape: (Timepoints,)
        
        # Simple epoching: Find where marker changes from 0 to > 0
        diff_marker = np.diff(np.pad(marker, (1, 0), constant_values=0))
        event_onsets = np.where(diff_marker > 0)[0]
        
        # Extract 2-second windows (assuming 200Hz or 1000Hz, we'll extract 400 timepoints for now)
        # We will transpose so final shape is (Trials, Channels, Timepoints)
        window_size = 656 # Match BCI 2a size for consistency across models
        X_list, y_list = [], []
        
        for onset in event_onsets:
            if onset + window_size < len(data):
                trial_data = data[onset : onset + window_size, :]
                X_list.append(trial_data.T) # Transpose to (Channels, Timepoints)
                y_list.append(marker[onset])
                
        X_all = np.array(X_list, dtype=np.float32)
        
        # Map labels to 0-3 range (if there are more than 4 fingers, clip or group them)
        y_all = np.array(y_list) - 1 
        y_all = np.clip(y_all, 0, 3).astype(int) # Keep exactly 4 classes for model compatibility
        kaya_labels = np.array(["thumb", "index", "middle", "ring"])
        y_all = kaya_labels[y_all]
        
        split_idx = int(len(X_all) * 0.7)
        return X_all[:split_idx], y_all[:split_idx], X_all[split_idx:], y_all[split_idx:]
        
    # -----------------------------------------------------------------
    # 5. WAY-EEG-GAL (Grasp and Lift)
    # -----------------------------------------------------------------
    elif dataset_name == "WayEEGGAL":
        import pandas as pd
        import glob
        
        # Notice the nested train/train folder from the Kaggle extraction
        kaggle_dir = os.path.join(os.path.dirname(data_dir), "dataset", "grasp-and-lift-eeg-detection", "train", "train")
        print(f"Loading Grasp and Lift dataset from {kaggle_dir}...")
        
        # Fetching all series for the specific subject
        data_files = sorted(glob.glob(os.path.join(kaggle_dir, f"subj{subject_id}_series*_data.csv")))
        event_files = sorted(glob.glob(os.path.join(kaggle_dir, f"subj{subject_id}_series*_events.csv")))
        
        if not data_files:
            raise FileNotFoundError(f"No Kaggle CSV files found for subject {subject_id} in {kaggle_dir}")
            
        print(f"Found {len(data_files)} series for subject {subject_id}. Parsing CSVs...")
        
        X_list, y_list = [], []
        window_size = 656 # Match BCI 2a size (500Hz sampling rate, so this is ~1.3 seconds)
        
        # 6 Event types in order
        event_names = ['HandStart', 'FirstDigitTouch', 'BothStartLoadPhase', 'LiftOff', 'Replace', 'BothReleased']
        
        # We only process the first 2 series to save RAM during the demo
        for d_file, e_file in zip(data_files[:2], event_files[:2]):
            # Read CSVs and drop the string 'id' column
            df_data = pd.read_csv(d_file).drop(columns=['id']).values
            df_events = pd.read_csv(e_file).drop(columns=['id']).values
            
            # Find exact moments (onsets) where an event switches from 0 to 1
            diff_events = np.diff(np.pad(df_events, ((1, 0), (0, 0)), constant_values=0), axis=0)
            
            for event_class_idx in range(6):
                # Get row indices where this specific event started
                onsets = np.where(diff_events[:, event_class_idx] > 0)[0]
                
                for onset in onsets:
                    if onset + window_size < len(df_data):
                        trial_data = df_data[onset : onset + window_size, :]
                        X_list.append(trial_data.T) # Transpose to (Channels, Timepoints)
                        y_list.append(event_class_idx)
                        
        X_all = np.array(X_list, dtype=np.float32)
        y_all = np.array(y_list)
        
        # Limit to 4 classes to perfectly match the Transformer/MiniRocket architecture
        valid_indices = y_all < 4
        X_all = X_all[valid_indices]
        y_all = y_all[valid_indices].astype(int)
        
        way_labels = np.array(["hand_start", "first_digit_touch", "both_start_load", "lift_off"])
        y_all = way_labels[y_all]
        
        split_idx = int(len(X_all) * 0.7)
        return X_all[:split_idx], y_all[:split_idx], X_all[split_idx:], y_all[split_idx:]

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

if __name__ == "__main__":
    # Test the MOABB loaders
    for ds in ["BNCI2014_001", "PhysionetMI", "HighGamma"]:
        try:
            X_tr, y_tr, X_te, y_te = load_dataset(ds, subject_id=1)
            print(f"{ds} Loaded -> X_train shape: {X_tr.shape}, y_train shape: {y_tr.shape}\n")
        except Exception as e:
            print(f"Error loading {ds}: {e}\n")
