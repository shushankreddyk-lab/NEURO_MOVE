import os
import sys
import mne
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.minirocket_engine import MiniRocketPipeline
from src.cnn_lstm_engine import CNN_LSTM_Pipeline
from src.preprocessing import preprocess_pipeline

def get_data_for_runs(runs, class_mapping):
    data_path = r"D:\dataset\data"
    mne.set_config('MNE_DATASETS_EEGBCI_PATH', data_path)
    
    all_X = []
    all_y = []
    
    # Iterate over all 109 subjects
    for subject in range(1, 110):
        try:
            raw_fnames = mne.datasets.eegbci.load_data(subject, runs, path=data_path, update_path=False)
            raws = [mne.io.read_raw_edf(f, preload=True) for f in raw_fnames]
            raw = mne.concatenate_raws(raws)
            mne.datasets.eegbci.standardize(raw)
            
            target_channels = ['FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4', 'C1', 'C2', 'C5', 'C6']
            available_channels = raw.ch_names
            picked_channels = [ch for ch in target_channels if ch in available_channels]
            if len(picked_channels) > 0:
                raw.pick_channels(picked_channels)
            
            raw.apply_function(lambda x: x * 1e6)
            events, _ = mne.events_from_annotations(raw)
            
            # Map events using class_mapping
            mapped_events = []
            for ev in events:
                if ev[2] in class_mapping:
                    ev_copy = ev.copy()
                    ev_copy[2] = class_mapping[ev[2]]
                    mapped_events.append(ev_copy)
            
            if len(mapped_events) == 0:
                continue
                
            mapped_events = np.array(mapped_events)
            
            # We construct a dummy event_dict based on what's present
            unique_ids = np.unique(mapped_events[:, 2])
            event_dict = {str(uid): uid for uid in unique_ids}
            
            X, y = preprocess_pipeline(raw, mapped_events, event_dict)
            all_X.append(X)
            all_y.append(y)
            print(f"Successfully processed subject {subject}")
        except Exception as e:
            print(f"Skipping subject {subject} due to error: {e}")
    
    if not all_X:
        return np.array([]), np.array([])
        
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)

def main():
    print("Loading motor imagery for Fists (L/R)...")
    # In Physionet, events from annotations gives T0=1, T1=2, T2=3 by default
    # Wait, mne.events_from_annotations without event_id maps sequentially.
    # T0 is rest, T1 is task 1, T2 is task 2.
    # Usually, mne maps T0: 1, T1: 2, T2: 3
    # Let's verify by just using standard mapping.
    
    # We will just load them normally to inspect what mne returns
    tmp_raw_fnames = mne.datasets.eegbci.load_data(1, [4], path=os.path.abspath('data/raw'), update_path=False)
    tmp_raw = mne.io.read_raw_edf(tmp_raw_fnames[0], preload=True)
    tmp_events, tmp_event_dict = mne.events_from_annotations(tmp_raw)
    print("Default Event Dict:", tmp_event_dict)
    
    t1_id = tmp_event_dict.get('T1', 2)
    t2_id = tmp_event_dict.get('T2', 3)
    
    print("Loading Fists imagery...")
    X_fists, y_fists = get_data_for_runs([4, 8, 12], {t1_id: 0, t2_id: 1})
    
    print("Loading Feet/Both Fists imagery...")
    X_feet, y_feet = get_data_for_runs([6, 10, 14], {t1_id: 2, t2_id: 3})
    
    if len(X_fists) > 0 and len(X_feet) > 0:
        X = np.concatenate([X_fists, X_feet], axis=0)
        y = np.concatenate([y_fists, y_feet], axis=0)
    else:
        print("No valid data generated.")
        return

    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    print(f"Total dataset: {X.shape}, Labels: {np.unique(y)}")
    
    print("Training MiniRocket...")
    mr = MiniRocketPipeline(num_kernels=10000)
    mr.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    mr.save('models/minirocket.pkl')
    print("Saved models/minirocket.pkl")
    
    print("Training CNN-LSTM...")
    cnn = CNN_LSTM_Pipeline(epochs=50, lr=1e-4, batch_size=32)
    cnn.fit(X, y)
    
    cnn.save('models/cnn_lstm.pth')
    print("Saved models/cnn_lstm.pth")

if __name__ == '__main__':
    main()
