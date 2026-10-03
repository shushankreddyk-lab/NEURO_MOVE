import os
import torch
import traceback
import threading
from src.eegnet_engine import EEGNet_Pipeline
from src.minirocket_engine import MiniRocketPipeline
from src.advanced_eeg_engine import AdvancedEEGPipeline

model_dir = os.path.join('d:\\eeg-minirocket-project\\eeg-mi-bci', 'models')

def load_model(mname):
    mpath = os.path.join(model_dir, mname)
    try:
        if 'eegnet' in mname:
            pipeline = EEGNet_Pipeline(num_classes=4, channels=64, samples=656)
        elif 'conformer' in mname:
            pipeline = AdvancedEEGPipeline(num_classes=4, channels=64, samples=656)
        else:
            pipeline = MiniRocketPipeline(in_channels=64, seq_len=656)
        pipeline.load(mpath)
        print(f"SUCCESS {mname} in thread {threading.get_ident()}")
    except Exception as e:
        print(f"ERROR {mname}: {e}")
        print(traceback.format_exc())

threads = []
for mname in [
    'master_gpu_minirocket_subs1to5_20261002_195856.pth',
    'master_eegnet_subs1to5_20261002_195810.pth',
    'master_conformer_subs1to5_20261002_195741.pth'
]:
    t = threading.Thread(target=load_model, args=(mname,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
