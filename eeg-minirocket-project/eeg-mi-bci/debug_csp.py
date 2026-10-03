import sys
sys.path.insert(0, r'D:\pip_packages')
import os
import numpy as np
import traceback

try:
    from src.csp_engine import CSP_Engine
except ModuleNotFoundError as exc:
    raise SystemExit("Skipping: legacy src.csp_engine was removed (%s)" % exc)

X = np.random.randn(1, 20, 656) # dummy data

model = CSP_Engine(classifier_type="lda", n_components=4)
print("\n--- Testing CSP ---")
try:
    model_path = None
    for root, dirs, files in os.walk(r'd:\eeg-minirocket-project\eeg-mi-bci\models'):
        for f in files:
            if 'csp' in f.lower():
                model_path = os.path.join(root, f)
                break
        if model_path: break
    
    if model_path and os.path.exists(model_path):
        print(f"Loading {model_path}...")
        model.load(model_path)
        probs = model.predict_proba(X)
        print(f"Success! Probs shape: {probs.shape}")
    else:
        print(f"No model found for CSP")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
