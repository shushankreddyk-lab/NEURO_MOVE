import os
import sys
import time
import numpy as np
import mne
import warnings

warnings.filterwarnings('ignore')

_D_PKGS = r"D:\pip_packages"
if _D_PKGS not in sys.path:
    sys.path.insert(0, _D_PKGS)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dataset_2a_loader import load_bci_2a_data
from eegnet_engine import EEGNet_Pipeline
from minirocket_engine import MiniRocketPipeline

def simulate_realtime(model_path, data_dir, model_type="EEGNet", subject=1, window_size=656, step_size=160):
    """
    Simulate a real-time BCI pipeline.
    Streams an evaluation file, buffering 656 samples (4.1s at 160Hz) at a time,
    overlapping by step_size (1s updates).
    """
    print(f"--- INITIALIZING BCI REAL-TIME SIMULATION ---")
    print(f"Loading Model: {model_path} ({model_type})")
    
    # 1. Load Model
    if model_type == "EEGNet":
        pipeline = EEGNet_Pipeline(epochs=1, batch_size=16) # dummy params
        pipeline.load(model_path)
    elif model_type == "MiniRocket":
        # Need to know num_kernels, in_channels, seq_len. Assuming 10000, 22, 656
        pipeline = MiniRocketPipeline(num_kernels=10000, in_channels=22, seq_len=656)
        pipeline.load(model_path)
    else:
        print(f"Model type {model_type} not supported for simulation yet.")
        return

    # 2. Load continuous data for streaming
    filename = f"A{subject:02d}E.gdf"
    filepath = os.path.join(data_dir, filename)
    if not os.path.exists(filepath):
        print(f"Evaluation file {filepath} not found.")
        return
        
    print(f"Loading continuous stream from: {filepath}")
    raw = mne.io.read_raw_gdf(filepath, preload=True, verbose=False)
    raw.pick_channels(raw.ch_names[:22])
    raw.set_eeg_reference('average', projection=False)
    raw.filter(4., 38., fir_design='firwin', skip_by_annotation='edge', verbose=False)
    
    if raw.info['sfreq'] != 160.0:
        raw.resample(160.0)
        
    continuous_data = raw.get_data() * 1e6 # Convert to uV
    num_channels, total_samples = continuous_data.shape
    
    print("Stream ready. Beginning real-time inference loop...")
    print("=" * 60)
    
    # Classes mapping
    classes = {0: "Left Hand", 1: "Right Hand", 2: "Both Feet", 3: "Tongue"}
    
    # Simulate streaming loop
    current_idx = 0
    predictions = []
    
    while current_idx + window_size <= total_samples:
        # Extract current window
        window = continuous_data[:, current_idx:current_idx+window_size]
        # Reshape to (1, channels, time) for inference
        X_batch = window[np.newaxis, ...]
        
        # Inference
        t0 = time.perf_counter()
        if hasattr(pipeline, 'predict'):
            # Some pipelines have predict method returning array of shape (n_samples,)
            y_pred = pipeline.predict(X_batch)
            pred_class = y_pred[0]
            confidence = 1.0 # default dummy if no proba
            if hasattr(pipeline, 'predict_proba'):
                try:
                    probs = pipeline.predict_proba(X_batch)
                    confidence = np.max(probs[0])
                except:
                    pass
        else:
            # Fallback if pipeline doesn't have predict implemented cleanly for inference
            # (assuming PyTorch underlying model)
            import torch
            pipeline.model.eval()
            with torch.no_grad():
                tensor_X = torch.tensor(X_batch, dtype=torch.float32).to(pipeline.device)
                if model_type == "EEGNet":
                    tensor_X = tensor_X.unsqueeze(1) # Add channel dim for EEGNet
                outputs = pipeline.model(tensor_X)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred_class = np.argmax(probs)
                confidence = np.max(probs)
                
        t1 = time.perf_counter()
        latency = (t1 - t0) * 1000
        
        timestamp_sec = (current_idx + window_size) / 160.0
        
        print(f"[{timestamp_sec:>6.2f}s] Pred: {classes.get(pred_class, 'Unknown'):<12} | Conf: {confidence*100:>5.1f}% | Latency: {latency:>5.2f}ms")
        
        # Advance by step_size (e.g. 1 second updates)
        current_idx += step_size
        
        # Just simulate 20 steps so we don't spam forever
        if current_idx > step_size * 20:
            break
            
        time.sleep(0.1) # slow down for visual effect

    print("=" * 60)
    print("Simulation Complete.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained .pth model")
    parser.add_argument("--model_type", type=str, default="EEGNet", help="Architecture (EEGNet, MiniRocket)")
    parser.add_argument("--data_dir", type=str, default=r"D:\eeg-minirocket-project\BCICIV_2a_gdf", help="Path to dataset")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID to stream")
    
    args = parser.parse_args()
    simulate_realtime(args.model_path, args.data_dir, args.model_type, args.subject)
