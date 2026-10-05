import subprocess
import sys

datasets = ["BNCI2014_001", "PhysionetMI", "HighGamma", "KayaFingers", "WayEEGGAL", "DREAMER"]

for dataset in datasets:
    print(f"===========================================================")
    print(f"Starting training for {dataset} on subjects 1-10 (ALL MODELS)")
    print(f"===========================================================")
    
    cmd = [
        sys.executable, "src/train_master.py",
        "--mode", "master",
        "--dataset", dataset,
        "--model", "all",
        "--sub_start", "1",
        "--sub_end", "10",
        "--epochs", "2"  # Keep it small just so it completes in a reasonable time if it's a test, 
                         # but wait, the default was 10. I will leave it default unless specified, but let's use default.
    ]
    
    # Remove epochs to use default
    cmd.pop()
    cmd.pop()
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Successfully finished {dataset}\n")
    except subprocess.CalledProcessError as e:
        print(f"Failed on {dataset}: {e}\n")
