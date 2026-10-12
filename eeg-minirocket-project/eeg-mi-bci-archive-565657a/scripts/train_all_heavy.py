import subprocess
import time
import sys

datasets = ["bnci2014_001", "highgamma", "kayafingers", "wayeeggal"]
models = ["MiniRocket", "Shallow ConvNet", "EEGNet", "Advanced Transformer", "CNN-LSTM"]

for dataset in datasets:
    print(f"\n{'='*50}\nStarting Dataset: {dataset}\n{'='*50}")
    for model in models:
        print(f"  Training {model} on {dataset}...")
        cmd = [
            sys.executable, "src/train_master.py",
            "--mode", "master",
            "--dataset", dataset,
            "--epochs", "300",
            "--top_channels", "20",
            "--model", model
        ]
        try:
            # Run sequentially
            process = subprocess.run(cmd, capture_output=True, text=True)
            if process.returncode != 0:
                print(f"    [!] Error training {model} on {dataset}:")
                # Print last few lines of error
                lines = process.stderr.strip().split('\n')
                print("      " + "\n      ".join(lines[-10:]))
            else:
                print(f"    [*] Successfully trained {model} on {dataset}!")
        except Exception as e:
            print(f"    [!] Exception: {e}")

print("\nALL HEAVY OVERFITTING TRAINING COMPLETED!")
