import sys
sys.path.insert(0, r'D:\pip_packages')
import os
import torch
import numpy as np

from src.advanced_eeg_engine import AdvancedEEGPipeline

X = np.random.randn(5, 20, 656)

pipeline = AdvancedEEGPipeline(num_classes=4, channels=20, samples=656)
model_path = r"d:\eeg-minirocket-project\eeg-mi-bci\models\master_cnn_lstm_subs1to76_20261001_063346.pth"
print("Loading model:", model_path)
pipeline.load(model_path)
print("feat_mean:", pipeline.feat_mean)
try:
    probs = pipeline.predict_proba(X)
    print("Success! Probs shape:", probs.shape)
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
