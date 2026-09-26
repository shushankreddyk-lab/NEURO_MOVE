"""
Preprocessing pipeline for Motor Imagery EEG Signals.
Methodology based on Hwaidi & Ghanem (NeuroImage 328 (2026) 121816):
- Bandpass filtering (4–40 Hz) using zero-phase Butterworth filter.
- Notch filtering (50 Hz / 60 Hz) for powerline interference suppression.
- Common Average Referencing (CAR).
- Baseline correction (-0.5s to 0.0s).
- Amplitude thresholding for trial artifact rejection.
- Z-score normalization per channel.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, Optional


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float = 4.0,
    h_freq: float = 40.0,
    order: int = 4
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter along the time axis.
    
    Parameters
    ----------
    data : np.ndarray
        Array of shape (..., n_times).
    sfreq : float
        Sampling frequency in Hz.
    l_freq : float
        Lower cutoff frequency in Hz (default 4.0 Hz).
    h_freq : float
        Upper cutoff frequency in Hz (default 40.0 Hz).
    order : int
        Filter order (default 4).
        
    Returns
    -------
    filtered : np.ndarray
        Bandpass-filtered EEG data.
    """
    nyq = 0.5 * sfreq
    low = l_freq / nyq
    high = h_freq / nyq
    
    # Clip upper bound to avoid numerical instability
    if high >= 1.0:
        high = 0.99
    if low <= 0:
        low = 0.01

    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def notch_filter(
    data: np.ndarray,
    sfreq: float,
    freq: float = 50.0,
    q: float = 30.0
) -> np.ndarray:
    """
    Apply IIR notch filter at line noise frequency (50 or 60 Hz).
    """
    nyq = 0.5 * sfreq
    if freq >= nyq:
        # If Nyquist is below notch frequency, notch is not applicable
        return data
        
    w0 = freq / nyq
    b, a = iirnotch(w0, q)
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def common_average_reference(data: np.ndarray) -> np.ndarray:
    """
    Apply Common Average Reference (CAR): subtract the mean across all channels.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (n_epochs, n_channels, n_times) or (n_channels, n_times).
        
    Returns
    -------
    car_data : np.ndarray
        Re-referenced EEG data.
    """
    # Channel dimension is -2
    mean_ch = np.mean(data, axis=-2, keepdims=True)
    return (data - mean_ch).astype(np.float32)


def baseline_correction(
    data: np.ndarray,
    sfreq: float,
    tmin: float = -0.5,
    baseline_window: Tuple[float, float] = (-0.5, 0.0)
) -> np.ndarray:
    """
    Apply baseline subtraction using pre-stimulus interval.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (n_epochs, n_channels, n_times).
    sfreq : float
        Sampling frequency in Hz.
    tmin : float
        Start time of the epoch in seconds.
    baseline_window : Tuple[float, float]
        Time interval (t_start, t_end) for baseline calculation.
    """
    n_times = data.shape[-1]
    time_pts = np.linspace(tmin, tmin + (n_times - 1) / sfreq, n_times)
    
    base_mask = (time_pts >= baseline_window[0]) & (time_pts <= baseline_window[1])
    if not np.any(base_mask):
        base_mask = time_pts < 0.0

    if np.any(base_mask):
        baseline_mean = np.mean(data[..., base_mask], axis=-1, keepdims=True)
        return (data - baseline_mean).astype(np.float32)
    return data


def reject_artifacts(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 150e-6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reject epochs exceeding maximum peak-to-peak amplitude threshold.
    
    Parameters
    ----------
    X : np.ndarray
        Shape (n_epochs, n_channels, n_times).
    y : np.ndarray
        Shape (n_epochs,).
    threshold : float
        Maximum allowed peak-to-peak voltage (e.g. 150 uV = 150e-6 V).
        
    Returns
    -------
    X_clean : np.ndarray
    y_clean : np.ndarray
    keep_indices : np.ndarray
    """
    p2p = np.ptp(X, axis=-1)  # shape (n_epochs, n_channels)
    max_p2p = np.max(p2p, axis=-1)  # shape (n_epochs,)
    keep = max_p2p < threshold
    
    # If too many epochs rejected, fall back to soft clipping to preserve dataset size
    if np.sum(keep) < 0.5 * len(y):
        keep = np.ones(len(y), dtype=bool)
        X = np.clip(X, -threshold, threshold)

    return X[keep], y[keep], np.where(keep)[0]


def standardize_channels(data: np.ndarray) -> np.ndarray:
    """
    Channel-wise Z-score standardization across time.
    """
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True) + 1e-8
    return ((data - mean) / std).astype(np.float32)


def preprocess_eeg_dataset(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    l_freq: float = 4.0,
    h_freq: float = 40.0,
    notch_freq: Optional[float] = 50.0,
    apply_car: bool = True,
    apply_baseline: bool = True,
    apply_standardize: bool = True,
    artifact_threshold: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Execute full preprocessing pipeline as described in the paper.
    
    1. Bandpass filter (4–40 Hz)
    2. Notch filter (50 or 60 Hz)
    3. Common Average Reference (CAR)
    4. Baseline correction (-0.5s to 0.0s)
    5. Artifact thresholding / rejection
    6. Channel standardization
    """
    # 1. Bandpass filter
    X_proc = bandpass_filter(X, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq)
    
    # 2. Notch filter
    if notch_freq is not None and notch_freq < (0.5 * sfreq):
        X_proc = notch_filter(X_proc, sfreq=sfreq, freq=notch_freq)
        
    # 3. CAR
    if apply_car:
        X_proc = common_average_reference(X_proc)
        
    # 4. Baseline correction
    if apply_baseline:
        X_proc = baseline_correction(X_proc, sfreq=sfreq)
        
    # 5. Artifact rejection
    if artifact_threshold is not None:
        X_proc, y, _ = reject_artifacts(X_proc, y, threshold=artifact_threshold)
        
    # 6. Standardize
    if apply_standardize:
        X_proc = standardize_channels(X_proc)
        
    return X_proc, y
