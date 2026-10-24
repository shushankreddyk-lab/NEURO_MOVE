"""
Synthetic EEG Data Generator for CI/Testing.

Generates realistic motor imagery EEG data without requiring real datasets.
"""
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import json


# Standard 10-20 motor cortex channels
MOTOR_CHANNELS = [
    'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4',
    'Cz', 'FCz', 'CPz', 'F3', 'F4', 'Fz',
    'P3', 'P4', 'Pz', 'O1', 'O2', 'Oz',
    'T3', 'T4'
]

# PhysioNet-style 64 channels (subset)
PHYSIONET_64 = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2',
    'Fpz', 'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6',
    'FC3', 'FC4', 'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4',
    'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'FT7', 'FT8', 'TP7', 'TP8'
]

CLASS_LABELS = {
    'physionet': ['left_fist', 'right_fist', 'both_fists', 'both_feet'],
    'bci2a': ['left_hand', 'right_hand', 'both_feet', 'tongue'],
    'highgamma': ['left_hand', 'right_hand', 'both_feet', 'rest'],
    'kaya': ['thumb', 'index', 'middle', 'ring', 'little'],
    'way': ['handstart', 'firstdigittouch', 'bothstartloadphase', 'liftoff', 'replace', 'bothreleased'],
    'nemar': ['thumb', 'index', 'middle', 'ring', 'little'],
}


def generate_mu_rhythm(n_channels: int, n_samples: int, sfreq: float,
                       mu_freq: float = 10.0, alpha_power: float = 10.0) -> np.ndarray:
    """Generate synthetic mu/alpha rhythm (8-12 Hz) over motor cortex."""
    t = np.arange(n_samples) / sfreq
    data = np.zeros((n_channels, n_samples))
    
    for ch in range(n_channels):
        # Base mu rhythm with channel-specific phase
        phase = np.random.uniform(0, 2*np.pi)
        mu = alpha_power * np.sin(2*np.pi*mu_freq*t + phase)
        
        # Add beta rhythm (18-26 Hz)
        beta_freq = np.random.uniform(18, 26)
        beta = 3.0 * np.sin(2*np.pi*beta_freq*t + phase + np.pi/4)
        
        # Pink noise (1/f)
        noise = np.random.randn(n_samples)
        # Simple 1/f approximation
        freqs = np.fft.rfftfreq(n_samples, 1/sfreq)
        fft_noise = np.fft.rfft(noise)
        fft_noise[1:] /= np.sqrt(freqs[1:])
        pink = np.fft.irfft(fft_noise, n=n_samples)
        
        data[ch] = mu + beta + 2.0 * pink
    
    return data


def generate_erd_pattern(n_channels: int, n_samples: int, sfreq: float,
                         active_channels: List[int], erd_strength: float = 0.6,
                         onset: float = 0.5, duration: float = 3.0) -> np.ndarray:
    """Generate Event-Related Desynchronization (ERD) pattern."""
    t = np.arange(n_samples) / sfreq
    mask = np.zeros(n_samples)
    onset_idx = int(onset * sfreq)
    dur_idx = int(duration * sfreq)
    mask[onset_idx:onset_idx+dur_idx] = 1.0
    # Smooth edges
    from scipy.signal import windows
    taper = windows.hann(min(100, dur_idx//4))
    mask[onset_idx:onset_idx+len(taper)] *= taper
    mask[onset_idx+dur_idx-len(taper):onset_idx+dur_idx] *= taper[::-1]
    
    data = np.zeros((n_channels, n_samples))
    for ch in active_channels:
        # ERD: power reduction in mu/beta during motor imagery
        data[ch] = -erd_strength * mask * np.random.uniform(0.8, 1.2)
    
    return data


def generate_synthetic_eeg(n_trials: int = 100,
                           n_channels: int = 20,
                           n_samples: int = 656,
                           sfreq: float = 160.0,
                           n_classes: int = 4,
                           dataset: str = 'physionet',
                           snr_db: float = 3.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Generate synthetic MI-EEG data with realistic ERD/ERS patterns.
    
    Returns:
        X: (n_trials, n_channels, n_samples)
        y: (n_trials,) class labels
        channel_names: list of channel names
    """
    # Select channel names based on dataset
    if dataset == 'bci2a':
        ch_names = PHYSIONET_64[:22]
    elif dataset == 'highgamma':
        ch_names = [f'CH{i:03d}' for i in range(128)]
    elif n_channels <= 20:
        ch_names = MOTOR_CHANNELS[:n_channels]
    else:
        ch_names = [f'EEG_{i:03d}' for i in range(n_channels)]
    
    # Adjust n_channels to match ch_names
    n_channels = len(ch_names)
    
    # Class-specific active channels (motor cortex lateralization)
    class_channel_map = {
        0: [i for i, ch in enumerate(ch_names) if '3' in ch or 'C3' in ch or 'FC3' in ch or 'CP3' in ch],  # Left hand
        1: [i for i, ch in enumerate(ch_names) if '4' in ch or 'C4' in ch or 'FC4' in ch or 'CP4' in ch],  # Right hand
        2: [i for i, ch in enumerate(ch_names) if 'z' in ch.lower() or 'Cz' in ch],  # Both feet/hands
        3: [i for i, ch in enumerate(ch_names) if 'F' in ch or 'P' in ch],  # Tongue/Rest
    }
    
    X = np.zeros((n_trials, n_channels, n_samples), dtype=np.float32)
    y = np.zeros(n_trials, dtype=int)
    
    samples_per_class = n_trials // n_classes
    
    for cls in range(n_classes):
        start_idx = cls * samples_per_class
        end_idx = start_idx + samples_per_class if cls < n_classes - 1 else n_trials
        n_cls_trials = end_idx - start_idx
        
        active_ch = class_channel_map.get(cls, [])
        
        for trial in range(n_cls_trials):
            # Base EEG with mu/beta rhythms
            base = generate_mu_rhythm(n_channels, n_samples, sfreq)
            
            # Add ERD for active class channels
            erd = generate_erd_pattern(n_channels, n_samples, sfreq, active_ch, 
                                       erd_strength=0.7, onset=0.5, duration=3.0)
            
            # Add ERS (synchronization) for contralateral channels
            contra_ch = class_channel_map.get((cls + 1) % 2, []) if cls < 2 else []
            ers = generate_erd_pattern(n_channels, n_samples, sfreq, contra_ch,
                                       erd_strength=-0.3, onset=0.5, duration=3.0)
            
            # Combine with noise scaled by SNR
            noise_power = np.mean(base**2) / (10**(snr_db/10))
            noise = np.random.randn(n_channels, n_samples) * np.sqrt(noise_power)
            
            X[start_idx + trial] = base + erd + ers + noise
            y[start_idx + trial] = cls
    
    # Shuffle
    idx = np.random.permutation(n_trials)
    X = X[idx]
    y = y[idx]
    
    return X, y, ch_names


def generate_continuous_eeg(duration_sec: float = 60.0,
                            n_channels: int = 20,
                            sfreq: float = 160.0,
                            event_interval: float = 4.0,
                            dataset: str = 'physionet') -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    """Generate continuous EEG with periodic MI events."""
    n_samples = int(duration_sec * sfreq)
    X, _, ch_names = generate_synthetic_eeg(
        n_trials=int(duration_sec / event_interval),
        n_channels=n_channels,
        n_samples=int(event_interval * sfreq),
        sfreq=sfreq,
        dataset=dataset
    )
    
    # Concatenate trials
    continuous = X.reshape(-1, n_channels)
    
    # Create events array
    n_events = X.shape[0]
    events = np.zeros((n_events, 3), dtype=int)
    events[:, 0] = np.arange(n_events) * int(event_interval * sfreq)
    events[:, 2] = np.arange(n_events) % 4 + 1  # 1-4 class codes
    
    event_id = {f'class_{i+1}': i+1 for i in range(4)}
    
    return continuous, events, ch_names, event_id


def save_synthetic_dataset(output_dir: str,
                           dataset_name: str = 'synthetic_physionet',
                           n_subjects: int = 10,
                           trials_per_subject: int = 200,
                           **kwargs):
    """Save synthetic dataset to NPZ files (one per subject)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for subj in range(1, n_subjects + 1):
        X, y, ch_names = generate_synthetic_eeg(
            n_trials=trials_per_subject,
            dataset=dataset_name.replace('synthetic_', ''),
            **kwargs
        )
        
        npz_path = output_path / f"{dataset_name}_sub{subj:03d}.npz"
        np.savez_compressed(npz_path,
                            X=X, y=y, ch_names=ch_names,
                            sfreq=kwargs.get('sfreq', 160.0),
                            dataset=dataset_name)
        print(f"Saved {npz_path} ({X.shape})")


if __name__ == "__main__":
    # Quick test
    X, y, ch_names = generate_synthetic_eeg(n_trials=50, n_channels=20, n_samples=656)
    print(f"Generated: X={X.shape}, y={y.shape}, classes={np.unique(y)}")
    print(f"Channels: {ch_names[:5]}...")
    
    # Test continuous
    cont, events, ch_names, event_id = generate_continuous_eeg(duration_sec=10.0)
    print(f"Continuous: {cont.shape}, events: {events.shape}")
    print(f"Event IDs: {event_id}")