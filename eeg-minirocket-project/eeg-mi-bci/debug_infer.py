import sys
sys.path.insert(0, r'D:\pip_packages')
import os
import torch
import numpy as np
import traceback

from src.advanced_eeg_engine import AdvancedEEGPipeline
from src.minirocket_engine import MiniRocketPipeline
from src.eegnet_engine import EEGNet_Pipeline
from src.convnets_engine import ConvNet_Pipeline

X = np.random.randn(1, 20, 656) # dummy data

models = {
    "CNN-LSTM": AdvancedEEGPipeline(num_classes=4, channels=20, samples=656),
    "MiniRocket": MiniRocketPipeline(in_channels=20, seq_len=656),
    "EEGNet": EEGNet_Pipeline(num_classes=4, channels=20, samples=656),
    "Shallow": ConvNet_Pipeline(arch="shallow", num_classes=4, channels=20, samples=656),
    "Deep": ConvNet_Pipeline(arch="deep", num_classes=4, channels=20, samples=656)
}

for name, model in models.items():
    print(f"\n--- Testing {name} ---")
    try:
        model_path = None
        for root, dirs, files in os.walk(r'd:\eeg-minirocket-project\eeg-mi-bci\models'):
            for f in files:
                if (name.lower() in f.lower() or 
                    (name == 'Shallow' and 'shallow' in f.lower()) or
                    (name == 'Deep' and 'deep' in f.lower()) or
                    (name == 'MiniRocket' and 'minirocket' in f.lower()) or
                    (name == 'EEGNet' and 'eegnet' in f.lower())):
                    model_path = os.path.join(root, f)
                    break
            if model_path: break
        
        if model_path and os.path.exists(model_path):
            print(f"Loading {model_path}...")
            model.load(model_path)
            probs = model.predict_proba(X)
            print(f"Success! Probs shape: {probs.shape}")
        else:
            print(f"No model found for {name}")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
