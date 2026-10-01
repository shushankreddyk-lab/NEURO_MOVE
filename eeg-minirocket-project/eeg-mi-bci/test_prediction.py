import sys
import numpy as np
from src.advanced_eeg_engine import AdvancedEEGPipeline

try:
    print('Testing AdvancedEEGPipeline prediction...')
    pipe1 = AdvancedEEGPipeline(num_classes=4, channels=20, samples=656)
    pipe1.feat_mean = np.random.randn(1, 20, 656)
    pipe1.feat_std = np.ones((1, 20, 656))
    
    import torch
    
    X = np.random.randn(2, 20, 656)
    res = pipe1.predict_proba(X)
    print('AdvancedEEGPipeline OK:', res.shape)
    
except Exception as e:
    print(f'Error: {e}')
