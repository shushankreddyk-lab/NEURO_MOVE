"""
Unit and integration test suite for EEG Motor Imagery classification pipeline.
"""

import sys
import os
import unittest
import numpy as np

# Add workspace root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eeg-mi-bci")))

from src.minirocket_engine import MiniRocketPipeline
from src.cnn_lstm_engine import CNN_LSTM_Pipeline

class TestEEGPipeline(unittest.TestCase):

    def setUp(self):
        self.n_epochs = 20
        self.n_channels = 16
        self.n_times = 256
        self.sfreq = 128.0
        self.n_classes = 4

        self.X = np.random.randn(self.n_epochs, self.n_channels, self.n_times).astype(np.float32)
        self.y = np.random.randint(0, self.n_classes, size=(self.n_epochs,))
        self.ch_names = [f"EEG_{i}" for i in range(self.n_channels)]

    def test_minirocket_pipeline(self):
        """Test MiniRocket training and prediction."""
        mr = MiniRocketPipeline(num_kernels=200, in_channels=self.n_channels, seq_len=self.n_times, head_epochs=2)
        mr.sfreq = self.sfreq
        mr.channel_names = self.ch_names
        mr.fit(self.X, self.y)
        
        preds = mr.predict(self.X)
        self.assertEqual(len(preds), self.n_epochs)
        
        probs = mr.predict_proba(self.X)
        self.assertEqual(probs.shape, (self.n_epochs, self.n_classes))

    def test_cnn_lstm_pipeline(self):
        """Test CNN-LSTM pipeline."""
        pipeline = CNN_LSTM_Pipeline(channels=self.n_channels, samples=self.n_times, num_classes=self.n_classes, epochs=2, batch_size=8)
        pipeline.fit(self.X, self.y)
        preds = pipeline.predict(self.X)
        self.assertEqual(len(preds), self.n_epochs)

if __name__ == "__main__":
    unittest.main()
