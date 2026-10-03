import glob
import os
import pickle
import numpy as np
import sklearn
print("scikit-learn version:", sklearn.__version__)
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
candidates = sorted(glob.glob(os.path.join(model_dir, "*csp*.pkl")))
if not candidates:
    raise SystemExit("Skipping: no CSP .pkl checkpoints found in " + model_dir)
model_path = candidates[0]
print("Loading model:", model_path)
with open(model_path, "rb") as handle:
    pipeline = pickle.load(handle)
print("Loaded pipeline:", pipeline)
X = np.random.randn(5, 20, 656)
try:
    probs = pipeline.predict_proba(X)
    print("Success! Probs shape:", probs.shape)
except Exception:
    import traceback
    traceback.print_exc()
