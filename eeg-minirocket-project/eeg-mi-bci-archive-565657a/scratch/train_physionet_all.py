import os

models = ["MiniRocket", "EEGNet", "Shallow", "CNN-LSTM", "Conformer"]
dataset = "physionet"

for model in models:
    print(f"\n======================================")
    print(f"Training {model} on {dataset}...")
    print(f"======================================\n")
    os.system(f'python src/train_master.py --mode master --dataset {dataset} --model "{model}" --epochs 300 --top_channels 20')
