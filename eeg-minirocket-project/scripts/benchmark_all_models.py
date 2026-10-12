import subprocess
import json
import time

models = ["MiniRocket", "CNN-LSTM", "EEGNet", "Shallow ConvNet", "Deep ConvNet", "CSP + LDA"]
dataset_dir = r"d:\eeg-minirocket-project\physionet"

print("==================================================")
print("     BCI MODEL COMPARISON (QUICK BENCHMARK)       ")
print("==================================================")
print("Running Subjects 67 to 76 | 100 Epochs per Neural Net\n")

results = {}

for m in models:
    print(f"Testing {m}...")
    start_time = time.time()
    cmd = [
        "python", "src/train_master.py",
        "--mode", "master",
        "--dataset", dataset_dir,
        "--model", m,
        "--epochs", "100",
        "--sub_start", "67",
        "--sub_end", "76"
    ]
    
    # Run the training
    process = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time
    
    # Parse final accuracy
    val_acc = "Error"
    train_acc = "Error"
    
    for line in process.stdout.split("\n"):
        try:
            if line.startswith("{"):
                data = json.loads(line)
                if data.get("type") == "epoch":
                    val_acc = f"{data.get('val_acc', 0):.2f}%"
                    train_acc = f"{data.get('train_acc', 0):.2f}%"
                elif data.get("type") == "complete":
                    # CSP + LDA might just spit out completion metrics if implemented
                    if val_acc == "Error":
                        val_acc = "Completed"
                        train_acc = "Completed"
        except Exception:
            pass
            
    # For models that don't output epoch JSON (like CSP+LDA which is closed-form)
    if "CSP" in m and val_acc == "Error":
        val_acc = "Check UI Logs"
        train_acc = "Check UI Logs"
            
    results[m] = {
        "Train Acc": train_acc,
        "Val Acc": val_acc,
        "Time (s)": f"{elapsed:.1f}s"
    }
    
    print(f" -> Val Acc: {val_acc} | Time: {elapsed:.1f}s\n")

print("==================================================")
print(f"{'Architecture':<20} | {'Train Acc':<12} | {'Val Acc':<12} | {'Time'}")
print("-" * 65)
for m, res in results.items():
    print(f"{m:<20} | {res['Train Acc']:<12} | {res['Val Acc']:<12} | {res['Time (s)']}")
print("==================================================")
