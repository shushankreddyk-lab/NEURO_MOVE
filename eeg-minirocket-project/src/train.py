"""
Training script for Motor Imagery EEG Classification (MiniRocket vs CNN-LSTM).
Reproducing Hwaidi & Ghanem (NeuroImage 328 (2026) 121816):
- Evaluates both MiniRocket (10k kernels + RidgeClassifierCV) and Hybrid CNN-LSTM
- Benchmarks accuracy, F1-score, training time, and inference latency
- Saves models to models/ and results to results/metrics/
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.data_loader import load_physionet_subject, load_bci_iv_2a_subject
from src.preprocessing import preprocess_eeg_dataset
from src.minirocket_pipeline import MiniRocketPipeline
from src.cnn_lstm_model import CNNLSTMPipeline


def run_training(
    dataset: str = "physionet",
    num_subjects: int = 10,
    num_kernels: int = 2000,
    epochs: int = 15,
    batch_size: int = 32,
    arch_type: str = "primary",
    output_dir: str = "results",
    model_dir: str = "models",
    random_state: int = 42
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_path / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"EEG Motor Imagery Classification Benchmark: {dataset.upper()}")
    print(f"Subjects: {num_subjects} | MiniRocket Kernels: {num_kernels} | CNN-LSTM Epochs: {epochs}")
    print("=" * 70)

    results_records = []

    for sub_id in range(1, num_subjects + 1):
        print(f"\n---> Processing Subject {sub_id}/{num_subjects}...")
        
        # 1. Load Data
        t_load_start = time.perf_counter()
        if dataset.lower() == "physionet":
            X_raw, y_raw, sfreq, ch_names = load_physionet_subject(
                subject_id=sub_id,
                downsample_sfreq=128.0,
                use_synthetic_fallback=False
            )
        else:
            X_raw, y_raw, sfreq, ch_names = load_bci_iv_2a_subject(
                subject_id=sub_id,
                use_synthetic_fallback=False
            )
        t_load = time.perf_counter() - t_load_start

        # 2. Preprocessing: 4-40Hz bandpass, CAR, baseline correction, z-score
        t_prep_start = time.perf_counter()
        X, y = preprocess_eeg_dataset(
            X_raw, y_raw, sfreq=sfreq,
            l_freq=4.0, h_freq=40.0,
            notch_freq=50.0,
            apply_car=True,
            apply_baseline=True,
            apply_standardize=True
        )
        t_prep = time.perf_counter() - t_prep_start

        # 80/20 stratified train/test split per subject
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=random_state, stratify=y
        )
        # 10% of train as validation for CNN-LSTM early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=random_state, stratify=y_train
        )

        n_channels, n_times = X.shape[1], X.shape[2]
        n_classes = len(np.unique(y))

        # -------------------------------------------------------------
        # 3. MiniRocket Training & Evaluation
        # -------------------------------------------------------------
        print(f"  [MiniRocket] Training {num_kernels} kernels + RidgeClassifierCV...")
        mr_pipeline = MiniRocketPipeline(
            num_kernels=num_kernels,
            max_dilations_per_kernel=28,
            random_state=random_state
        )
        mr_pipeline.fit(X_train, y_train)
        mr_preds = mr_pipeline.predict(X_test)
        mr_acc = float(accuracy_score(y_test, mr_preds))
        mr_f1 = float(f1_score(y_test, mr_preds, average='macro'))

        # Benchmark MiniRocket Latency on single trial
        single_trial = X_test[0:1]
        mr_latency = mr_pipeline.benchmark_latency(single_trial, n_warmup=3, n_runs=30)
        print(f"  [MiniRocket] Acc: {mr_acc*100:.2f}% | Macro F1: {mr_f1*100:.2f}% | Latency: {mr_latency['latency_mean_ms']:.2f} ms")

        # -------------------------------------------------------------
        # 4. CNN-LSTM Training & Evaluation
        # -------------------------------------------------------------
        print(f"  [CNN-LSTM] Training PyTorch {arch_type} architecture (lr=1e-5, wd=0.01)...")
        cnn_pipeline = CNNLSTMPipeline(
            in_channels=n_channels,
            n_classes=n_classes,
            arch_type=arch_type,
            lr=1e-5,
            weight_decay=0.01
        )
        cnn_pipeline.fit(
            X_tr, y_tr,
            X_val=X_val, y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            early_stopping_patience=5,
            verbose=False
        )
        cnn_preds = cnn_pipeline.predict(X_test)
        cnn_acc = float(accuracy_score(y_test, cnn_preds))
        cnn_f1 = float(f1_score(y_test, cnn_preds, average='macro'))

        # Benchmark CNN-LSTM Latency on single trial
        cnn_latency = cnn_pipeline.benchmark_latency(single_trial, n_warmup=3, n_runs=30)
        print(f"  [CNN-LSTM] Acc: {cnn_acc*100:.2f}% | Macro F1: {cnn_f1*100:.2f}% | Latency: {cnn_latency['latency_mean_ms']:.2f} ms")

        # Speedup ratio
        speedup = cnn_latency['latency_mean_ms'] / max(1e-5, mr_latency['latency_mean_ms'])
        print(f"  --> MiniRocket speedup over CNN-LSTM: {speedup:.1f}x faster inference!")

        # Record metrics
        record = {
            "subject": f"S{sub_id:02d}",
            "dataset": dataset,
            "n_channels": n_channels,
            "n_samples": len(X),
            "minirocket_acc": mr_acc,
            "minirocket_f1": mr_f1,
            "minirocket_train_time_s": mr_pipeline.train_time_sec_,
            "minirocket_latency_ms": mr_latency['latency_mean_ms'],
            "cnn_lstm_acc": cnn_acc,
            "cnn_lstm_f1": cnn_f1,
            "cnn_lstm_train_time_s": cnn_pipeline.train_time_sec_,
            "cnn_lstm_latency_ms": cnn_latency['latency_mean_ms'],
            "speedup_factor": speedup
        }
        results_records.append(record)

        # Save model checkpoints for Subject 1 as reference
        if sub_id == 1:
            mr_pipeline.save(model_path / f"minirocket_{dataset}_s01.joblib")
            cnn_pipeline.save(model_path / f"cnn_lstm_{dataset}_s01.pt")
            # Save test split for evaluation script and demo
            np.savez_compressed(
                model_path / f"sample_test_data_{dataset}.npz",
                X_test=X_test,
                y_test=y_test,
                ch_names=ch_names,
                sfreq=sfreq
            )

    df_results = pd.DataFrame(results_records)
    summary_path = metrics_dir / f"benchmark_summary_{dataset}.csv"
    df_results.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY (Hwaidi & Ghanem 2026 Reproduction)")
    print("=" * 70)
    print(df_results[[
        "subject", "minirocket_acc", "cnn_lstm_acc",
        "minirocket_latency_ms", "cnn_lstm_latency_ms", "speedup_factor"
    ]].to_string(index=False))

    avg_mr_acc = df_results["minirocket_acc"].mean() * 100
    avg_cnn_acc = df_results["cnn_lstm_acc"].mean() * 100
    avg_mr_lat = df_results["minirocket_latency_ms"].mean()
    avg_cnn_lat = df_results["cnn_lstm_latency_ms"].mean()
    avg_speedup = df_results["speedup_factor"].mean()

    print("-" * 70)
    print(f"Mean MiniRocket Accuracy: {avg_mr_acc:.2f}% (Target: ~98.6% on PhysioNet / ~92.6% on BCI 2a)")
    print(f"Mean CNN-LSTM Accuracy:   {avg_cnn_acc:.2f}% (Target: ~98.1% on PhysioNet / ~92.3% on BCI 2a)")
    print(f"Mean MiniRocket Latency:  {avg_mr_lat:.2f} ms")
    print(f"Mean CNN-LSTM Latency:    {avg_cnn_lat:.2f} ms")
    print(f"Mean MiniRocket Speedup:  {avg_speedup:.1f}x faster inference")
    print(f"Results successfully saved to: {summary_path}")
    print("=" * 70)
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MiniRocket vs CNN-LSTM for Motor Imagery EEG")
    parser.add_argument("--dataset", type=str, default="physionet", choices=["physionet", "bci_iv_2a"])
    parser.add_argument("--num_subjects", type=int, default=10, help="Number of subjects (default: 10)")
    parser.add_argument("--num_kernels", type=int, default=2000, help="MiniRocket kernels (default: 2000)")
    parser.add_argument("--epochs", type=int, default=10, help="CNN-LSTM training epochs (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--arch_type", type=str, default="primary", choices=["primary", "table1"])
    args = parser.parse_args()

    run_training(
        dataset=args.dataset,
        num_subjects=args.num_subjects,
        num_kernels=args.num_kernels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        arch_type=args.arch_type
    )
