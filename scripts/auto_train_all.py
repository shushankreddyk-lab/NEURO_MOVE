from pathlib import Path

# --- NeuroMove path resolution (env-overridable) ---
_PROJECT_ROOT = Path(os.environ.get("NEUROMOVE_ROOT", Path(__file__).resolve().parents[1]))
_PIP_PACKAGES = os.environ.get("NEUROMOVE_PIP_PACKAGES")
import subprocess
import os
import sys

# Paths
python_exe = sys.executable
script_path = os.path.join("eeg-mi-bci", "src", "train_master.py")
physionet_path = os.path.join(_PROJECT_ROOT, "datasets/physionet")
bci_path = os.path.join(_PROJECT_ROOT, "datasets/BCICIV_2a_gdf")

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
        '--dataset_name', prefix,
        '--mode', 'master',
        '--epochs', '80',
        '--lr', '0.001',
        '--kernels', '5000',
        '--partition', '80',
        '--sub_start', str(start),
        '--sub_end', str(end)
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = _PIP_PACKAGES
    
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

# Train on NEMAR
nemar_path = os.path.join(_PROJECT_ROOT, "datasets/nemar_on008446")
nemar_batches = [(1, 1)] # testing 1 subject first
for start, end in nemar_batches:
    run_training(nemar_path, start, end, "NEMAR Finger MI")

print("ALL TRAINING JOBS SUCCESSFULLY COMPLETED!")

print("Pushing updates to git...")
try:
    import datetime
    commit_msg = f"Automated training update: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    subprocess.run(["git", "add", "-f", "eeg-mi-bci/artifacts/training_results.csv"], cwd=r"D:\eeg-minirocket-project", check=True)
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=r"D:\eeg-minirocket-project", check=True)
    subprocess.run(["git", "push"], cwd=r"D:\eeg-minirocket-project", check=True)
    print("Successfully pushed training results to git!")
except Exception as e:
    print(f"Error pushing to git: {e}")
