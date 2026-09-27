import os
import sys
import numpy as np
import mne
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from binary_parser import load_local_eeg_data
from preprocessing import apply_car, apply_bandpass_filter

def plot_alpha_block(results_dir):
    """
    Plots PSD of O1, O2, OZ for Eyes Open (R01) vs Eyes Closed (R02).
    Expects alpha (8-13 Hz) power to increase in R02 (Eyes Closed).
    """
    print("Computing Occipital Alpha Block...")
    raws, _, _ = load_local_eeg_data(1, [1, 2])
    
    if len(raws) < 2:
        print("Missing R01 or R02")
        return
        
    r01_raw, r02_raw = raws[0], raws[1]
    
    # Preprocess
    r01_raw = apply_bandpass_filter(apply_car(r01_raw), 4, 38)
    r02_raw = apply_bandpass_filter(apply_car(r02_raw), 4, 38)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot PSD for R01
    r01_raw.compute_psd(fmin=4, fmax=38, picks=['O1', 'O2', 'OZ']).plot(axes=axes[0], show=False)
    axes[0].set_title('Eyes Open (R01)')
    
    # Plot PSD for R02
    r02_raw.compute_psd(fmin=4, fmax=38, picks=['O1', 'O2', 'OZ']).plot(axes=axes[1], show=False)
    axes[1].set_title('Eyes Closed (R02)')
    
    plt.tight_layout()
    os.makedirs(results_dir, exist_ok=True)
    plt.savefig(os.path.join(results_dir, 'alpha_block.png'))
    plt.close()

def plot_motor_erd(results_dir):
    """
    Plots PSD of C3 vs C4 for Left vs Right tasks.
    """
    print("Computing Contralateral Motor ERD...")
    # R03/R07/R11 (real), R04/R08/R12 (imagined)
    # We will just take R04 (left fist imagined, right fist imagined)
    raws, events_list, mappings = load_local_eeg_data(1, [4])
    
    if len(raws) == 0:
        return
        
    raw = raws[0]
    events = events_list[0]
    mapping = mappings[0]
    
    raw = apply_bandpass_filter(apply_car(raw), 4, 38)
    
    # Epoching
    # T1 = left fist imagined, T2 = right fist imagined
    event_id = {}
    for k, v in mapping.items():
        if 'left' in v:
            event_id['left_fist'] = k
        elif 'right' in v:
            event_id['right_fist'] = k
            
    if 'left_fist' not in event_id or 'right_fist' not in event_id:
        print("Missing events")
        return
        
    epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4.0, baseline=None, preload=True)
    
    # Compute PSD for each condition
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs['left_fist'].compute_psd(fmin=4, fmax=38, picks=['C3', 'C4']).plot(axes=axes[0], show=False, average=True)
    axes[0].set_title('Left Fist Imagined')
    
    epochs['right_fist'].compute_psd(fmin=4, fmax=38, picks=['C3', 'C4']).plot(axes=axes[1], show=False, average=True)
    axes[1].set_title('Right Fist Imagined')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'motor_erd.png'))
    plt.close()

if __name__ == "__main__":
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results', 'subject_analysis'))
    plot_alpha_block(results_dir)
    plot_motor_erd(results_dir)
    print(f"Plots saved to {results_dir}")
