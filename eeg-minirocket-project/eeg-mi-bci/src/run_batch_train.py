import os
import sys
import subprocess
import time

def run_training(dataset_name, dataset_path, model_name, sub_start=1, sub_end=5):
    print(f"\n{'='*50}\nStarting {model_name} Training on {dataset_name} (Subjects {sub_start}-{sub_end})\n{'='*50}")
    
    script_path = os.path.join(os.path.dirname(__file__), 'train_master.py')
    
    cmd = [
        sys.executable,
        script_path,
        "--mode", "master",
        "--dataset", dataset_path,
        "--model", model_name,
        "--epochs", "40", # increased for higher accuracy
        "--lr", "1e-3",
        "--kernels", "15000", # increased kernels for MiniRocket robustness
        "--partition", "80",
        "--sub_start", str(sub_start),
        "--sub_end", str(sub_end)
    ]
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            # Print only important lines to not spam stdout
            if "Epoch" in line or "Validation" in line or "accuracy" in line.lower() or "completed" in line.lower():
                print(line.strip())
        process.wait()
        
        if process.returncode != 0:
            print(f"Error during {dataset_name} ({model_name}) training.")
        else:
            print(f"SUCCESS: {dataset_name} with {model_name} completed!")
            
    except Exception as e:
        print(f"Failed to run {dataset_name} ({model_name}): {e}")

if __name__ == "__main__":
    datasets = {
        "DREAMER": r"D:\eeg-minirocket-project\dataset\DREAMER.mat",
        "HIGH-GAMMA": r"D:\eeg-minirocket-project\dataset\pone.0218181.s001",
        "WAY-EEG-GAL": r"D:\eeg-minirocket-project\dataset\grasp-and-lift-eeg-detection",
        "BCI Competition IV 2a": r"D:\eeg-minirocket-project\dataset\bci_competition",
        "PhysioNet": r"D:\eeg-minirocket-project\physionet",
        "KAYA (Finger Movements)": r"D:\eeg-minirocket-project\dataset"
    }
    
    models = ["MiniRocket", "CNN-LSTM", "EEGNet", "Shallow ConvNet", "Advanced Transformer"]
    
    for d_name, d_path in datasets.items():
        for m_name in models:
            run_training(d_name, d_path, m_name)
