import glob
import os
import torch
import numpy as np
from src.advanced_eeg_engine import AdvancedEEGPipeline
X = np.random.randn(1, 64, 481)
pipeline = AdvancedEEGPipeline(num_classes=4, channels=64, samples=481)
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
candidates = sorted(glob.glob(os.path.join(model_dir, "*conformer*.pth")))
candidates += sorted(glob.glob(os.path.join(model_dir, "*.pth")))
if not candidates:
    raise SystemExit("Skipping: no .pth checkpoints found in " + model_dir)
model_path = candidates[0]
print("Loading model:", model_path)
pipeline.load(model_path)
print("feat_mean:", pipeline.feat_mean)
try:
    probs = pipeline.predict_proba(X)
    print("Success! Probs shape:", probs.shape)
except Exception:
    import traceback
    print("Error during predict_proba")
    traceback.print_exc()
