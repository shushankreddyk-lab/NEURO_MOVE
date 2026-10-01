import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eeg-mi-bci")))

from src.minirocket_engine import MiniRocketPipeline
from src.cnn_lstm_engine import CNN_LSTM_Pipeline
from src.dataset_loader_all import load_dataset
from src.preprocessing import apply_bandpass_filter, apply_car
import numpy as np
import time

try:
    X_raw, y, ch_names = generate_synthetic_eeg(n_epochs=2, n_channels=64, n_times=576, sfreq=128.0)
    X_proc, y = preprocess_eeg_dataset(X_raw, y, sfreq=128.0)
    
    mr = MiniRocketPipeline(num_kernels=200, random_state=42)
    mr.fit(X_proc, y)
    
    cnn = CNNLSTMPipeline(in_channels=64, n_classes=4, arch_type="primary", device="cpu")
    cnn.fit(X_proc, y, epochs=1, batch_size=2, verbose=False)

    print("Models fitted. Testing inference logic from app.py...")

    single_trial = X_proc[0:1]
    
    t0 = time.perf_counter()
    mr_pred = mr.predict(single_trial)[0]
    mr_probs = mr.predict_proba(single_trial)[0]
    
    print(f"MR Pred: {mr_pred}, type: {type(mr_pred)}")
    print(f"MR Probs: {mr_probs}, shape: {mr_probs.shape}")

    cnn_pred = cnn.predict(single_trial)[0]
    cnn_probs = cnn.predict_proba(single_trial)[0]

    print(f"CNN Pred: {cnn_pred}, type: {type(cnn_pred)}")
    print(f"CNN Probs: {cnn_probs}, shape: {cnn_probs.shape}")
    print("Success!")
except Exception as e:
    import traceback
    traceback.print_exc()
