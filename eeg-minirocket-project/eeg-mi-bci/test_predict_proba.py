import sys
sys.path.insert(0, 'D:\\pip_packages')
import os
import torch
import numpy as np

from src.advanced_eeg_engine import AdvancedEEGPipeline
from src.minirocket_engine import MiniRocketPipeline
from src.eegnet_engine import EEGNet_Pipeline
from src.convnets_engine import ConvNet_Pipeline
from src.cnn_lstm_engine import CNN_LSTM_Pipeline

MODEL_DIR = 'd:\\eeg-minirocket-project\\eeg-mi-bci\\models'
MODELS = (
    ("Conformer", AdvancedEEGPipeline(num_classes=4, channels=64, samples=481), ("conformer",)),
    ("CNN-LSTM", CNN_LSTM_Pipeline(num_classes=4, channels=64, samples=481), ("cnn_lstm",)),
    ("MiniRocket", MiniRocketPipeline(in_channels=64, seq_len=481), ("minirocket",)),
    ("EEGNet", EEGNet_Pipeline(num_classes=4, channels=64, samples=481), ("eegnet",)),
    ("Shallow", ConvNet_Pipeline(arch="shallow", num_classes=4, channels=64, samples=481), ("shallow",)),
)

def find_checkpoint(keys):
    for fname in sorted(os.listdir(MODEL_DIR)):
        lowered = fname.lower()
        if fname.endswith(".pth") and any(key in lowered for key in keys):
            return os.path.join(MODEL_DIR, fname)
    return None

for name, model, keys in MODELS:
    print("Testing %s..." % name)
    try:
        model_path = find_checkpoint(keys)
        if model_path is None:
            print("  Skipped: no checkpoint available.")
            continue
        model.load(model_path)
        print("  Loaded checkpoint: %s" % os.path.basename(model_path))
        probs = model.predict_proba(np.random.randn(1, 64, 481))
        print("  Probs shape: %s" % (probs.shape,))
    except Exception as exc:
        import traceback
        print("  Error: %s" % exc)
        traceback.print_exc()
