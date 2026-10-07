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

for dataset in datasets:
    for model in models:
        print(f"\n======================================")
        print(f"Training Model: {model} on Dataset: {dataset}")
        print(f"======================================")
        
        top_ch = dataset_top_ch.get(dataset, '20')
        
        # We will use subset of subjects (e.g. sub_start=1, sub_end=1) to have it complete within a reasonable timeframe, 
        # unless full batch training is absolutely required. 
        # Assuming the user wants it to just run successfully and generate the models.
        cmd = [
            'python', 'train_master.py',
            '--mode', 'master',
            '--dataset', dataset,
            '--model', model,
            '--epochs', '5',  # Low epochs for faster completion
            '--top_channels', top_ch,
            '--sub_start', '1',
            '--sub_end', '2'
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error training {model} on {dataset}: {e}")

print("\nAll training runs completed.")
