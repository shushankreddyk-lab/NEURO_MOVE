import sys
sys.path.insert(0, r'D:\pip_packages')
import pickle
import numpy as np
import sklearn
import mne
print("scikit-learn version:", sklearn.__version__)

with open(r'd:\eeg-minirocket-project\eeg-mi-bci\models\master_csp_lda_subs67to76_20260930_224256.pkl', 'rb') as f:
    pipeline = pickle.load(f)

print("Loaded pipeline:", pipeline)
X = np.random.randn(5, 20, 656)
try:
    probs = pipeline.predict_proba(X)
    print("Success! Probs shape:", probs.shape)
except Exception as e:
    import traceback
    traceback.print_exc()
