"""
Unit and integration test suite for EEG Motor Imagery classification pipeline.
Verifies:
1. Data loading and synthetic EEG generation
2. Preprocessing transformations (Bandpass, Notch, CAR, Baseline, Standardization)
3. MiniRocket feature extraction and RidgeClassifierCV inference
4. Hybrid CNN-LSTM and Table 1 PyTorch models (forward and backward passes)
5. Latency benchmarking functions
"""

import sys
import os
import unittest
import numpy as np
import torch

# Add workspace root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import generate_synthetic_eeg, PHYSIONET_CLASSES
from src.preprocessing import (
    bandpass_filter,
    notch_filter,
    common_average_reference,
    baseline_correction,
    standardize_channels,
    preprocess_eeg_dataset
)
from src.minirocket_pipeline import MiniRocketPipeline
from src.cnn_lstm_model import HybridCNNLSTM, Table1CNNLSTM, CNNLSTMPipeline


class TestEEGPipeline(unittest.TestCase):

    def setUp(self):
        self.n_epochs = 20
        self.n_channels = 16
        self.n_times = 256
        self.sfreq = 128.0
        self.n_classes = 4

        self.X, self.y, self.ch_names = generate_synthetic_eeg(
            n_epochs=self.n_epochs,
            n_channels=self.n_channels,
            n_times=self.n_times,
            sfreq=self.sfreq,
            n_classes=self.n_classes,
            random_state=42
        )

    def test_synthetic_data_generation(self):
        """Test dimensions and values of generated EEG signals."""
        self.assertEqual(self.X.shape, (self.n_epochs, self.n_channels, self.n_times))
        self.assertEqual(self.y.shape, (self.n_epochs,))
        self.assertEqual(len(self.ch_names), self.n_channels)
        self.assertTrue(np.all(np.isin(self.y, [0, 1, 2, 3])))

    def test_preprocessing_transforms(self):
        """Test signal filtering, CAR, and standardization."""
        # 1. Bandpass filter
        X_bp = bandpass_filter(self.X, sfreq=self.sfreq, l_freq=4.0, h_freq=40.0)
        self.assertEqual(X_bp.shape, self.X.shape)
        self.assertFalse(np.isnan(X_bp).any())

        # 2. CAR (Common Average Reference)
        X_car = common_average_reference(X_bp)
        # Mean across channel dimension should be near 0
        channel_means = np.mean(X_car, axis=1)
        np.testing.assert_allclose(channel_means, 0.0, atol=1e-6)

        # 3. Baseline correction
        X_base = baseline_correction(X_car, sfreq=self.sfreq, tmin=-0.5, baseline_window=(-0.5, 0.0))
        self.assertEqual(X_base.shape, self.X.shape)

        # 4. Standardization
        X_std = standardize_channels(X_base)
        stds = np.std(X_std, axis=-1)
        np.testing.assert_allclose(stds, 1.0, atol=1e-2)

        # 5. Full pipeline
        X_proc, y_proc = preprocess_eeg_dataset(self.X, self.y, sfreq=self.sfreq)
        self.assertEqual(X_proc.shape, self.X.shape)
        self.assertEqual(len(y_proc), len(self.y))

    def test_minirocket_pipeline(self):
        """Test MiniRocket training, prediction, and latency benchmarking."""
        mr = MiniRocketPipeline(num_kernels=200, random_state=42)
        mr.fit(self.X, self.y)
        self.assertTrue(mr.is_fitted)

        preds = mr.predict(self.X)
        self.assertEqual(len(preds), self.n_epochs)

        probs = mr.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.n_epochs, self.n_classes))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0, atol=1e-4)

        score = mr.score(self.X, self.y)
        self.assertGreater(score, 0.25)

        # Latency benchmark test
        bench = mr.benchmark_latency(self.X[0], n_warmup=2, n_runs=5)
        self.assertIn("latency_mean_ms", bench)
        self.assertGreater(bench["latency_mean_ms"], 0.0)

    def test_cnn_lstm_models(self):
        """Test forward and backward pass for both primary and Table 1 CNN-LSTM."""
        batch_size = 4
        x_tensor = torch.tensor(self.X[:batch_size], dtype=torch.float32)

        # 1. Primary Model
        model_primary = HybridCNNLSTM(in_channels=self.n_channels, n_classes=self.n_classes)
        out_primary = model_primary(x_tensor)
        self.assertEqual(out_primary.shape, (batch_size, self.n_classes))

        # Backward pass
        loss = out_primary.sum()
        loss.backward()
        self.assertIsNotNone(model_primary.conv1.weight.grad)

        # 2. Table 1 Model
        model_table1 = Table1CNNLSTM(in_channels=self.n_channels, n_classes=self.n_classes)
        out_table1 = model_table1(x_tensor)
        self.assertEqual(out_table1.shape, (batch_size, self.n_classes))

        # 3. Pipeline test
        pipeline = CNNLSTMPipeline(in_channels=self.n_channels, n_classes=self.n_classes, device="cpu")
        pipeline.fit(self.X, self.y, epochs=2, batch_size=8, verbose=False)
        preds = pipeline.predict(self.X)
        self.assertEqual(len(preds), self.n_epochs)

        bench = pipeline.benchmark_latency(self.X[0], n_warmup=2, n_runs=5)
        self.assertIn("latency_mean_ms", bench)
        self.assertGreater(bench["latency_mean_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
