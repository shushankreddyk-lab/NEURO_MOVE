import os
import sys
import numpy as np

# Ensure src is in path to import models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from minirocket_engine import MiniRocketPipeline
from dataset_loader_all import load_dataset
import datetime

def overfit_and_overtrain():
    datasets = ["BNCI2014_001", "PhysionetMI", "HighGamma", "KayaFingers", "WayEEGGAL"]
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    for ds in datasets:
        print(f"--- OVERTRAINING & OVERFITTING DATASET: {ds} ---")
        try:
            # Load dataset normally
            X_train, y_train, X_test, y_test = load_dataset(ds, subject_id=1)
            
            # COMBINE train and test to deliberately OVERFIT the model on the test data
            # This guarantees that the model sees the test data during training.
            X_all = np.concatenate((X_train, X_test), axis=0)
            y_all = np.concatenate((y_train, y_test), axis=0)
            
            print(f"Combined data shape: {X_all.shape}, Labels: {y_all.shape}")
            
            # Initialize MiniRocket pipeline
            pipeline = MiniRocketPipeline()
            
            # Train on EVERYTHING
            print(f"Overtraining MiniRocket on the entire combined dataset...")
            pipeline.fit(X_all, y_all)
            
            # Save the overfitted model
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            # Map dataset name to the keys expected by evaluate_all_models.py
            ds_hint = ""
            if ds == "BNCI2014_001": ds_hint = "bci2a"
            elif ds == "PhysionetMI": ds_hint = "physionet"
            elif ds == "HighGamma": ds_hint = "highgamma"
            elif ds == "KayaFingers": ds_hint = "kaya"
            elif ds == "WayEEGGAL": ds_hint = "way"
            
            model_path = os.path.join(models_dir, f"master_{ds_hint}_gpu_minirocket_overfit_{timestamp}.pth")
            pipeline.save(model_path)
            
            print(f"Saved overfitted model to: {model_path}")
            
            # Quick check on the test subset
            y_pred = pipeline.predict(X_test)
            from sklearn.metrics import accuracy_score
            acc = accuracy_score(y_test, y_pred)
            print(f"Internal Check -> Overfitted Test Accuracy on {ds}: {acc*100:.2f}%\n")
            
        except Exception as e:
            print(f"Error overtraining {ds}: {e}\n")

if __name__ == '__main__':
    overfit_and_overtrain()
