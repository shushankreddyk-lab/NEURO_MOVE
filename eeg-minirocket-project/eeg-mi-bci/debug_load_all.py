import os
import torch
import traceback

from src.advanced_eeg_engine import AdvancedEEGPipeline
from src.minirocket_engine import MiniRocketPipeline
from src.eegnet_engine import EEGNet_Pipeline
from src.convnets_engine import ConvNet_Pipeline
from src.cnn_lstm_engine import CNN_LSTM_Pipeline

model_dir = os.path.join('d:\\eeg-minirocket-project\\eeg-mi-bci', 'models')
model_names = [
    'master_gpu_minirocket_subs1to5_20261002_195856.pth',
    'master_eegnet_subs1to5_20261002_195810.pth',
    'master_conformer_subs1to5_20261002_195741.pth',
    'master_shallow_subs1to5_20261002_195823.pth',
    'master_cnn_lstm_subs1to5_20261002_195733.pth'
]

for model_name in model_names:
    model_path = os.path.join(model_dir, model_name)
    _n_ch = 64
    target_samples = 656
    _n_classes = 4

    try:
        if "conformer" in model_name.lower():
            pipeline = AdvancedEEGPipeline(num_classes=_n_classes, channels=_n_ch, samples=target_samples)
        elif "cnn_lstm" in model_name.lower():
            pipeline = CNN_LSTM_Pipeline(num_classes=_n_classes, channels=_n_ch, samples=target_samples)
        elif "minirocket" in model_name.lower():
            pipeline = MiniRocketPipeline(in_channels=_n_ch, seq_len=target_samples) # MiniRocket uses internal ridge
        elif "eegnet" in model_name.lower():
            pipeline = EEGNet_Pipeline(num_classes=_n_classes, channels=_n_ch, samples=target_samples)
        elif "shallow" in model_name.lower():
            pipeline = ConvNet_Pipeline(arch="shallow", num_classes=_n_classes, channels=_n_ch, samples=target_samples)
        else:
            continue
        
        pipeline.load(model_path)
        print(f"SUCCESS loading {model_name}")
    except Exception as e:
        print(f"ERROR loading {model_name}: {type(e).__name__} - {e}")
        print(traceback.format_exc())
