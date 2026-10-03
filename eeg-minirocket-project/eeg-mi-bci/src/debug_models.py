import os
import sys
import traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        import torch
        import numpy as np
        from cnn_lstm_engine import CNN_LSTM_Pipeline
        from eegnet_engine import EEGNet_Pipeline
        X = np.random.randn(10, 14, 128).astype(np.float32)
        y = np.random.randint(0, 3, size=(10,))
        print("Testing CNN-LSTM...")
        try:
            pipeline = CNN_LSTM_Pipeline(num_classes=3, channels=14, samples=128, epochs=1)
            pipeline.fit(X, y, X, y)
            print("CNN-LSTM success!")
        except Exception:
            print("CNN-LSTM Error:")
            traceback.print_exc()
        print()
        print("Testing EEGNet...")
        try:
            pipeline2 = EEGNet_Pipeline(num_classes=3, channels=14, samples=128, epochs=1)
            pipeline2.fit(X, y, X, y)
            print("EEGNet success!")
        except Exception:
            print("EEGNet Error:")
            traceback.print_exc()
    except Exception as exc:
        print("Import error: " + str(exc))

if __name__ == "__main__":
    main()
