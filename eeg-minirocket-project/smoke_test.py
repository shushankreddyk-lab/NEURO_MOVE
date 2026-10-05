import sys
import os
import numpy as np

# Add eeg-mi-bci to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "eeg-mi-bci")))

from src.minirocket_engine import MiniRocketPipeline

def main():
    print("Initializing MiniRocketPipeline...")
    # Initialize pipeline
    mr = MiniRocketPipeline(num_kernels=200, in_channels=22, seq_len=500, head_epochs=5)
    
    # Create dummy data
    X = np.random.randn(20, 22, 500).astype(np.float32)
    y = np.random.randint(0, 4, size=(20,))
    
    print("Fitting MiniRocketPipeline...")
    mr.fit(X, y)
    
    print("Predicting with MiniRocketPipeline...")
    preds = mr.predict(X)
    probs = mr.predict_proba(X)
    
    print(f"Predictions: {preds}")
    print(f"Probabilities shape: {probs.shape}")
    
    # Test uncertainty rejection logic from app.py
    for i, p in enumerate(probs):
        max_prob = np.max(p)
        pred_class = np.argmax(p)
        if max_prob < 0.60:
            print(f"Trial {i}: Uncertain (No Intent Detected) - max_prob: {max_prob:.4f}")
        else:
            print(f"Trial {i}: Class {pred_class} - max_prob: {max_prob:.4f}")

if __name__ == "__main__":
    main()
