import sys
sys.path.insert(0, r'D:\pip_packages')
import os
import torch
import numpy as np

from src.csp_engine import CSP_Engine

X = np.random.randn(5, 20, 656)

pipeline = CSP_Engine(classifier_type="lda", n_components=4)
model_path = r"d:\eeg-minirocket-project\eeg-mi-bci\models\master_csp_lda_subs67to76_20260930_224256.pkl"
print("Loading model:", model_path)
pipeline.load(model_path)
try:
    probs = pipeline.predict_proba(X)
    print("Success! Probs shape:", probs.shape)
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
