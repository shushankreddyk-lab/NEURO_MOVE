import os
import torch
import traceback
from src.eegnet_engine import EEGNet_Pipeline
from src.minirocket_engine import MiniRocketPipeline
from src.advanced_eeg_engine import AdvancedEEGPipeline

model_dir = os.path.join('d:\\eeg-minirocket-project\\eeg-mi-bci', 'models')

for mname in [
    'master_gpu_minirocket_subs1to5_20261002_195856.pth',
    'master_eegnet_subs1to5_20261002_195810.pth',
    'master_conformer_subs1to5_20261002_195741.pth'
]:
    mpath = os.path.join(model_dir, mname)
    try:
        if 'eegnet' in mname:
            p = EEGNet_Pipeline(num_classes=4, channels=64, samples=656)
        elif 'conformer' in mname:
            p = AdvancedEEGPipeline(num_classes=4, channels=64, samples=656)
        else:
            p = MiniRocketPipeline(in_channels=64, seq_len=656)
            
        p.load(mpath)
        print(f"SUCCESS: {mname}")
    except Exception as e:
        print(f"ERROR: {mname}")
        print(traceback.format_exc())
