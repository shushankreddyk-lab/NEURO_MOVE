import os
import glob
import mne
import numpy as np
import pandas as pd

def load_nemar_finger_data(dataset_path=r"D:\eeg-minirocket-project\dataset\NEMAR_on008446", subject="sub-01"):
    """
    Loads EDF and Event TSV files for the NEMAR Finger Movement Dataset.
    """
    eeg_dir = os.path.join(dataset_path, subject, "eeg")
    
    # Find all EDF files for the given subject
    edf_files = glob.glob(os.path.join(eeg_dir, "*.edf"))
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found for {subject} in {eeg_dir}")
        
    all_raws = []
    all_events = []
    
    # Finger classes mapped to 3-7 from dataset based on user request
    event_id = {'Thumb': 3, 'Index': 4, 'Middle': 5, 'Ring': 6, 'Little': 7}
    
    for edf_file in edf_files:
        print(f"Loading {edf_file}...")
        raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
        
        # Load corresponding TSV event file
        tsv_file = edf_file.replace('_eeg.edf', '_events.tsv')
        if os.path.exists(tsv_file):
            events_df = pd.read_csv(tsv_file, sep='\t')
            # Create MNE events array from TSV
            # (Note: TSV onset is in seconds, convert to samples)
            sfreq = raw.info['sfreq']
            events = np.zeros((len(events_df), 3), dtype=int)
            events[:, 0] = (events_df['onset'].values * sfreq).astype(int)
            
            # Map trial types to integers directly if they are 3 to 7
            for i, trial_type in enumerate(events_df['trial_type']):
                try:
                    tt_int = int(trial_type)
                    if 3 <= tt_int <= 7:
                        events[i, 2] = tt_int
                    else:
                        events[i, 2] = -1
                except ValueError:
                    # In case some strings exist
                    if 'thumb' in str(trial_type).lower(): events[i, 2] = 3
                    elif 'index' in str(trial_type).lower(): events[i, 2] = 4
                    elif 'middle' in str(trial_type).lower(): events[i, 2] = 5
                    elif 'ring' in str(trial_type).lower(): events[i, 2] = 6
                    elif 'little' in str(trial_type).lower(): events[i, 2] = 7
                    else: events[i, 2] = -1

            
            # Keep only valid finger events
            events = events[events[:, 2] > 0]
            
            all_raws.append(raw)
            all_events.append(events)

    if not all_raws:
        return None, None

    # Concatenate all runs
    raw_concat = mne.concatenate_raws(all_raws)
    
    current_offset = 0
    shifted_events = []
    for raw, evs in zip(all_raws, all_events):
        evs_shifted = evs.copy()
        evs_shifted[:, 0] += current_offset
        shifted_events.append(evs_shifted)
        # Add the number of samples in this raw file to the offset for the next one
        current_offset += raw.n_times
        
    events_concat = np.concatenate(shifted_events, axis=0)
    
    # Filter and Epoch
    raw_concat.filter(8., 30., fir_design='firwin') # Standard Alpha/Beta MI band
    epochs = mne.Epochs(raw_concat, events_concat, event_id, tmin=-0.5, tmax=2.0, 
                        baseline=(None, 0), preload=True, event_repeated='drop')
                        
    return epochs.get_data(), epochs.events[:, 2] - 3

if __name__ == "__main__":
    print("Testing NEMAR data loader...")
    try:
        X, y = load_nemar_finger_data()
        print(f"Loaded Shape: X={X.shape}, y={y.shape}")
        print("Ready for MiniRocket and Transformer models!")
    except Exception as e:
        print(f"Error (Make sure download is complete): {e}")
