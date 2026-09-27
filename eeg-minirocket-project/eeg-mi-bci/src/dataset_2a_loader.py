import os
import sys
import numpy as np
import mne

_D_PKGS = r"D:\pip_packages"
if _D_PKGS not in sys.path:
    sys.path.insert(0, _D_PKGS)

def load_bci_2a_data(data_dir, subjects, tmin=0.5, tmax=3.5, resample_freq=160.0):
    """
    Loads BCI Competition IV 2a dataset GDF files.
    
    Classes:
    1: Left Hand (769)
    2: Right Hand (770)
    3: Both Feet (771)
    4: Tongue (772)
    """
    X_all, y_all = [], []
    
    event_id = {'769': 0, '770': 1, '771': 2, '772': 3}
    
    for subject in subjects:
        # We load both Train (T) and Eval (E) files for the subject
        for suffix in ['T', 'E']:
            filename = f"A{subject:02d}{suffix}.gdf"
            filepath = os.path.join(data_dir, filename)
            
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue
                
            print(f"Loading {filepath}...")
            # Load raw data, ignore EOG channels, keep only 22 EEG channels
            raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
            
            # The first 22 channels are EEG
            eeg_channels = raw.ch_names[:22]
            raw.pick_channels(eeg_channels)
            
            # Bandpass filter
            raw.filter(4., 38., fir_design='firwin', skip_by_annotation='edge', verbose=False)
            
            # Extract events
            events, current_event_id = mne.events_from_annotations(raw, verbose=False)
            
            # Filter to only the 4 MI classes
            mi_events = []
            for ev in events:
                # Find the description for this event ID
                desc = None
                for d, i in current_event_id.items():
                    if i == ev[2]:
                        desc = d
                        break
                
                if desc in event_id:
                    # Append event with our standardized class ID (0 to 3)
                    mi_events.append([ev[0], ev[1], event_id[desc]])
                    
            if not mi_events:
                continue
                
            mi_events = np.array(mi_events)
            
            # Create Epochs
            epochs = mne.Epochs(
                raw, mi_events, 
                tmin=tmin, tmax=tmax, 
                baseline=None, 
                preload=True, 
                verbose=False
            )
            
            # Resample if needed
            if resample_freq and raw.info['sfreq'] != resample_freq:
                epochs.resample(resample_freq)
            
            # epochs.get_data() returns (trials, channels, times)
            # copy is important here
            X_all.append(epochs.get_data(copy=True) * 1e6)  # Convert to uV
            y_all.append(epochs.events[:, 2])
            
    if not X_all:
        return np.array([]), np.array([])
        
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    
    return X, y

if __name__ == "__main__":
    data_dir = r"D:\eeg-minirocket-project\BCICIV_2a_gdf"
    X, y = load_bci_2a_data(data_dir, [1], resample_freq=160.0)
    print(f"Loaded X shape: {X.shape}, y shape: {y.shape}")
