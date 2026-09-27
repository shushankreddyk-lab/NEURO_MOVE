import subprocess
import os
import sys

# Paths
python_exe = sys.executable
script_path = os.path.join("eeg-mi-bci", "src", "train_master.py")
physionet_path = r"D:\eeg-minirocket-project\physionet"
bci_path = r"D:\eeg-minirocket-project\BCICIV_2a_gdf"

# Define training batches to keep memory usage low but finish fast
physio_batches = [(1, 20), (21, 40), (41, 60), (61, 80), (81, 100), (101, 109)]
bci_batches = [(1, 5), (6, 9)]

def run_training(dataset_path, start, end, prefix):
    print(f"===========================================================")
    print(f"STARTING BATCH: {prefix} (Subjects {start}-{end})")
    print(f"===========================================================")
    args = [
        python_exe, script_path,
        '--dataset', dataset_path,
        '--mode', 'master',
        '--epochs', '50',      # Reduced epochs for faster completion
        '--lr', '0.001',
        '--kernels', '5000',   # Reduced kernels for faster completion
        '--partition', '80',
        '--sub_start', str(start),
        '--sub_end', str(end)
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = r'D:\pip_packages'
    
    process = subprocess.Popen(args, env=env)
    process.wait()
    print(f"BATCH COMPLETED: {prefix} (Subjects {start}-{end})\n")

print("COMMENCING FULL AUTOMATED TRAINING PIPELINE")

# Train on BCI
for start, end in bci_batches:
    run_training(bci_path, start, end, "BCI IV 2A")

# Train on PhysioNet
for start, end in physio_batches:
    run_training(physionet_path, start, end, "PhysioNet")

print("ALL TRAINING JOBS SUCCESSFULLY COMPLETED!")
