import os
import subprocess

datasets = ['PhysionetMI', 'BCI2a', 'HighGamma', 'WayEEGGAL', 'KayaFingers', 'NemarFingers', 'Dreamer']
models = ['MiniRocket', 'CNN-LSTM', 'Advanced Transformer', 'EEGNet', 'Shallow ConvNet']

dataset_top_ch = {
    'PhysionetMI': '64',
    'BCI2a': '22',
    'HighGamma': '40',
    'WayEEGGAL': '32',
    'KayaFingers': '16',
    'NemarFingers': '16',
    'Dreamer': '14'
}

print("Starting to train all models on all datasets (using subsets to speed up testing where applicable)...")

log_file_path = "training_console.log"
with open(log_file_path, "w") as f:
    f.write("=== ANTIGRAVITY / VS CODE TERMINAL LOG ===\n")
    f.write("Initializing distributed training pipeline...\n")

for dataset in datasets:
    for model in models:
        msg1 = f"\n======================================\n"
        msg2 = f"Training Model: {model} on Dataset: {dataset}\n"
        msg3 = f"======================================\n"
        print(msg1 + msg2 + msg3, end="")
        with open(log_file_path, "a") as f:
            f.write(msg1 + msg2 + msg3)
            f.flush()
        
        top_ch = dataset_top_ch.get(dataset, '20')
        
        cmd = [
            'python', 'src/train_master.py',
            '--mode', 'master',
            '--dataset', dataset,
            '--model', model,
            '--epochs', '100',  # Increased for massive accuracy
            '--top_channels', top_ch,
            '--sub_start', '1',
            '--sub_end', '5'  # Train on 5 subjects
        ]
        
        try:
            with open(log_file_path, "a") as f:
                subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            err_msg = f"Error training {model} on {dataset}: {e}\n"
            print(err_msg)
            with open(log_file_path, "a") as f:
                f.write(err_msg)

final_msg = "\nAll training runs completed.\n"
print(final_msg)
with open(log_file_path, "a") as f:
    f.write(final_msg)
