import os
import sys
import time
import unittest
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "eeg-mi-bci")))

from src.minirocket_engine import MiniRocketPipeline
from src.cnn_lstm_engine import CNN_LSTM_Pipeline
from src.advanced_eeg_engine import AdvancedEEGPipeline
from src.eegnet_engine import EEGNet_Pipeline
from src.convnets_engine import ConvNet_Pipeline
from src.preprocessing import apply_bandpass_filter, apply_car

class DemoLogicTest(unittest.TestCase):
    def test_pipelines_predict(self):
        rng = np.random.default_rng(42)
        trials, channels, samples = (12, 16, 256)
        X = rng.standard_normal((trials, channels, samples)).astype(np.float32)
        y = rng.integers(0, 4, size=(trials,))
        mr_pipeline = MiniRocketPipeline(num_kernels=200, in_channels=channels, seq_len=samples, head_epochs=2)
        mr_pipeline.sfreq = 128.0
        mr_pipeline.channel_names = ["EEG_" + str(i) for i in range(channels)]
        mr_pipeline.fit(X, y)
        mr_predictions = mr_pipeline.predict(X)
        mr_probabilities = mr_pipeline.predict_proba(X)
        self.assertEqual(len(mr_predictions), trials)
        self.assertEqual(mr_probabilities.shape, (trials, 4))
        cnn_pipeline = CNN_LSTM_Pipeline(channels=channels, samples=samples, num_classes=4, epochs=1, batch_size=4)
        cnn_pipeline.fit(X, y)
        cnn_predictions = cnn_pipeline.predict(X)
        cnn_probabilities = cnn_pipeline.predict_proba(X)
        self.assertEqual(len(cnn_predictions), trials)
        self.assertEqual(cnn_probabilities.shape, (trials, 4))
        
        # Test Advanced EEGPipeline
        adv_pipeline = AdvancedEEGPipeline(channels=channels, samples=samples, num_classes=4, epochs=1, batch_size=4)
        adv_pipeline.fit(X, y)
        adv_predictions = adv_pipeline.predict(X)
        adv_probabilities = adv_pipeline.predict_proba(X)
        self.assertEqual(len(adv_predictions), trials)
        self.assertEqual(adv_probabilities.shape, (trials, 4))
        
        # Test EEGNet Pipeline
        eegnet_pipeline = EEGNet_Pipeline(channels=channels, samples=samples, num_classes=4, epochs=1, batch_size=4)
        eegnet_pipeline.fit(X, y)
        eegnet_predictions = eegnet_pipeline.predict(X)
        eegnet_probabilities = eegnet_pipeline.predict_proba(X)
        self.assertEqual(len(eegnet_predictions), trials)
        self.assertEqual(eegnet_probabilities.shape, (trials, 4))
        
        # Test ShallowConvNet Pipeline
        convnet_pipeline = ConvNet_Pipeline(arch="shallow", channels=channels, samples=samples, num_classes=4, epochs=1, batch_size=4)
        convnet_pipeline.fit(X, y)
        convnet_predictions = convnet_pipeline.predict(X)
        convnet_probabilities = convnet_pipeline.predict_proba(X)
        self.assertEqual(len(convnet_predictions), trials)
        self.assertEqual(convnet_probabilities.shape, (trials, 4))

if __name__ == '__main__':
    unittest.main()
