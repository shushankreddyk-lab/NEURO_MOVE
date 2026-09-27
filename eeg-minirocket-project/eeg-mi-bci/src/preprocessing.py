import numpy as np
import mne
from scipy.signal import butter, filtfilt

def resample_data(raw, target_sfreq=160):
    """
    Ensure the data is at target_sfreq. PhysioNet is usually 160Hz natively.
    """
    if raw.info['sfreq'] != target_sfreq:
        raw.resample(target_sfreq)
    return raw

def apply_car(raw):
    """
    Apply Common Average Referencing (CAR) across all scalp electrodes.
    """
    raw.set_eeg_reference('average', projection=False)
    return raw

def spatial_channel_augmentation(raw):
    """
    Overridden for 10-channel architecture.
    Dynamically selects the 10 most active channels per subject based on signal variance,
    as performed in the base paper for subject-specific optimal channel selection.
    """
    # Calculate variance across time for each channel
    data = raw.get_data()
    variances = np.var(data, axis=1)
    
    # Get indices of the 10 channels with the highest variance
    top_10_indices = np.argsort(variances)[-10:][::-1]
    
    # Pick these channels by name to be safe with MNE
    ch_names = np.array(raw.ch_names)
    top_10_ch_names = ch_names[top_10_indices].tolist()
    
    raw.pick_channels(top_10_ch_names)
    return raw

def apply_bandpass_filter(raw, l_freq=4, h_freq=38):
    """
    Applies a 4th-order zero-phase Butterworth bandpass filter from 4 Hz to 38 Hz.
    """
    sfreq = raw.info['sfreq']
    nyq = sfreq / 2.0
    b, a = butter(4, [l_freq/nyq, h_freq/nyq], btype='bandpass')
    
    data = raw.get_data()
    filtered_data = filtfilt(b, a, data, axis=-1)
    
    raw_clean = mne.io.RawArray(filtered_data, raw.info, verbose=False)
    if len(raw.annotations) > 0:
        raw_clean.set_annotations(raw.annotations)
    return raw_clean

def epoch_and_segment(raw, events, event_id, tmin=0.0, tmax=4.1, group_id=None):
    """
    Epochs the data using a 4.1s window to yield exactly 656 samples at 160Hz.
    Explicitly filters out 'rest' (T0) events to focus strictly on active tasks,
    unless group_id == 1, which consists entirely of baseline resting events.
    """
    if group_id == 1:
        # Group 1 is just ocular baseline; keep it
        active_event_id = event_id
        
        # If there are no events in Group 1, generate fake continuous 2.0s windows
        if len(events) == 0:
            duration = raw.times[-1]
            window_samples = int(2.0 * raw.info['sfreq'])
            
            data = raw.get_data()
            n_windows = data.shape[1] // window_samples
            if n_windows == 0:
                return np.empty((0, len(raw.ch_names), window_samples)), np.empty(0)
                
            X_batch = np.zeros((n_windows, data.shape[0], window_samples))
            y_batch = np.zeros(n_windows) # class 0
            
            for i in range(n_windows):
                start = i * window_samples
                X_batch[i, :, :window_samples] = data[:, start:start+window_samples]
            
            return X_batch, y_batch
    else:
        # Explicit Rest Elimination for Groups 2-5
        active_event_id = {k: v for k, v in event_id.items() if 'rest' not in k.lower() and 'eyes' not in k.lower()}
    
    if len(active_event_id) == 0 or len(events) == 0:
        return np.empty((0, len(raw.ch_names), 656)), np.empty(0)
        
    tmax_adj = 4.1 - (1 / raw.info['sfreq'])
    epochs = mne.Epochs(raw, events, active_event_id, tmin, tmax_adj, baseline=None, preload=True, verbose=False)
    data = epochs.get_data() # (n_epochs, n_channels, n_times)
    labels = epochs.events[:, -1]
    
    target_samples = 656
    if data.shape[2] > target_samples:
        data = data[:, :, :target_samples]
    elif data.shape[2] < target_samples:
        pad_width = target_samples - data.shape[2]
        data = np.pad(data, ((0,0), (0,0), (0,pad_width)), mode='constant')
        
    return data, labels

def preprocess_pipeline(raw, events, event_id, group_id=None):
    raw = resample_data(raw, 160)
    raw = apply_car(raw)
    raw = apply_bandpass_filter(raw, 4, 38)
    raw = spatial_channel_augmentation(raw)
    X, y = epoch_and_segment(raw, events, event_id, group_id=group_id)
    return X, y

