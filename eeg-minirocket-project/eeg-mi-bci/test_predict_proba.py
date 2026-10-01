import sys
sys.path.insert(0, r'D:\pip_packages')
import os
import torch
import numpy as np

from src.advanced_eeg_engine import AdvancedEEGPipeline
from src.minirocket_engine import MiniRocketPipeline
from src.eegnet_engine import EEGNet_Pipeline
from src.convnets_engine import ConvNet_Pipeline
from src.csp_engine import CSP_Engine

X = np.random.randn(5, 20, 656) # 5 trials, 20 channels, 656 samples

models = {
    "CNN-LSTM": AdvancedEEGPipeline(num_classes=4, channels=20, samples=656),
    "MiniRocket": MiniRocketPipeline(in_channels=20, seq_len=656),
    "EEGNet": EEGNet_Pipeline(num_classes=4, channels=20, samples=656),
    "Shallow": ConvNet_Pipeline(arch="shallow", num_classes=4, channels=20, samples=656),
    "Deep": ConvNet_Pipeline(arch="deep", num_classes=4, channels=20, samples=656),
    "CSP": CSP_Engine(classifier_type="lda", n_components=4)
}

for name, model in models.items():
    print(f"Testing {name}...")
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            print(f"  Probs shape: {probs.shape}")
        else:
            print(f"  No predict_proba method!")
    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()
