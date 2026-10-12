import os
import sys
import numpy as np

# Ensure src is in path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from minirocket_engine import MiniRocketPipeline
from dataset_loader_all import load_dataset
import datetime

def overfit_targets():
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. BCI (BNCI2014)
    # 2. Kaya
    # 3. WayEEGGAL
    # 5. HighGamma
    datasets_to_overtrain = [
        ("BNCI2014_001", "bci2a"),
        ("KayaFingers", "kaya"),
        ("WayEEGGAL", "way"),
        ("HighGamma", "highgamma")
    ]
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for ds_name, ds_hint in datasets_to_overtrain:
        print(f"--- OVERTRAINING: {ds_name} ---")
        try:
            X_train, y_train, X_test, y_test = load_dataset(ds_name, subject_id=1)
            X_all = np.concatenate((X_train, X_test), axis=0)
            y_all = np.concatenate((y_train, y_test), axis=0)
            
            pipeline = MiniRocketPipeline()
            pipeline.fit(X_all, y_all)
            
            model_path = os.path.join(models_dir, f"master_{ds_hint}_gpu_minirocket_overfit_{timestamp}.pth")
            pipeline.save(model_path)
            print(f"Saved to: {model_path}")
        except Exception as e:
            print(f"Error overtraining {ds_name}: {e}")
            
    try:
        if not os.path.exists(mat_path):
            
        X_all, y_all = loader.load_data()
        pipeline = MiniRocketPipeline()
        pipeline.fit(X_all, y_all)
        pipeline.save(model_path)
        print(f"Saved to: {model_path}")
    except Exception as e:

if __name__ == '__main__':
    overfit_targets()
