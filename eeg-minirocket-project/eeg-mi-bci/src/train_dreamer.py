import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from dreamer_loader import DREAMERLoader
from advanced_eeg_engine import AdvancedEEGPipeline
from minirocket_engine import MiniRocketPipeline
from cnn_lstm_engine import CNN_LSTM_Pipeline

def main():
    print("=== DREAMER Dataset Emotion Recognition Training ===")
    
    # 1. Load Data
    mat_path = r"d:\eeg-minirocket-project\dataset\DREAMER.mat"
    loader = DREAMERLoader(mat_path, window_size_sec=1.0, overlap=0.5)
    X, y_raw = loader.load_data()
    
    # y_raw shape is (N, 3) -> [Valence, Arousal, Dominance]
    # Let's formulate a Binary Classification task for Arousal (1-5 scale)
    # High Arousal (>= 3) = 1, Low Arousal (< 3) = 0
    arousal_scores = y_raw[:, 1]
    y_arousal = (arousal_scores >= 3.0).astype(int)
    
    # Filter 
    # For now, we just pass the raw sliding windows (X is shape [N, 14, 128])
    n_samples, n_channels, n_timesteps = X.shape
    
    # 2. Chronological Split (No random shuffle to avoid temporal leakage)
    # Since windows are overlapping and contiguous, we just split chronologically.
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y_arousal[:split_idx], y_arousal[split_idx:]
    
    print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")
    print(f"Class Distribution (0: Low Arousal, 1: High Arousal):")
    print(f"Train: {np.bincount(y_train)}")
    print(f"Test: {np.bincount(y_test)}")
    
    # 3. Model Training - Advanced Transformer
    print("\n--- Training Advanced Transformer on DREAMER Arousal ---")
    transformer_model = AdvancedEEGPipeline(num_classes=2, channels=n_channels, samples=n_timesteps)
    transformer_model.fit(X_train, y_train, X_test, y_test)
    
    # Evaluate
    score_t = transformer_model.score(X_test, y_test)
    print(f"Transformer Test Accuracy (Arousal): {score_t * 100:.2f}%")
    
    # Save Model
    save_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    transformer_path = os.path.join(save_dir, "dreamer_transformer.pth")
    transformer_model.save(transformer_path, sfreq=loader.sfreq, channel_names=loader.ch_names)
    print(f"Saved to {transformer_path}")
    
    # 4. Model Training - MiniRocket
    print("\n--- Training MiniRocket on DREAMER Arousal ---")
    minirocket = MiniRocketPipeline(in_channels=n_channels, seq_len=n_timesteps)
    minirocket.fit(X_train, y_train)
    
    score_mr = minirocket.score(X_test, y_test)
    print(f"MiniRocket Test Accuracy (Arousal): {score_mr * 100:.2f}%")
    
    mr_path = os.path.join(save_dir, "dreamer_minirocket.pth")
    minirocket.save(mr_path, sfreq=loader.sfreq, channel_names=loader.ch_names)
    print(f"Saved to {mr_path}")

if __name__ == "__main__":
    main()
