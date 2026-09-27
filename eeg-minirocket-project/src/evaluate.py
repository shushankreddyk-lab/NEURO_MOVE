"""
Evaluation and visualization script reproducing figures and tables from
Hwaidi & Ghanem (NeuroImage 328 (2026) 121816):
- Confusion Matrices (Figures 6 & 7)
- ROC Curves & AUC (Figure 8)
- Per-subject Accuracy & F1 Comparison (Table 2 & Figure 9)
- Inference Latency Benchmark & Speedup Analysis
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
    f1_score
)
from sklearn.preprocessing import label_binarize

from src.minirocket_pipeline import MiniRocketPipeline
from src.cnn_lstm_model import CNNLSTMPipeline
from src.data_loader import PHYSIONET_CLASSES, BCI2A_CLASSES

# Publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def plot_confusion_matrices(
    y_true: np.ndarray,
    mr_preds: np.ndarray,
    cnn_preds: np.ndarray,
    class_names: list,
    save_path: Path
):
    """Plot side-by-side normalized confusion matrices for MiniRocket and CNN-LSTM."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cm_mr = confusion_matrix(y_true, mr_preds, normalize='true') * 100
    cm_cnn = confusion_matrix(y_true, cnn_preds, normalize='true') * 100

    # MiniRocket Confusion Matrix
    sns.heatmap(
        cm_mr, annot=True, fmt=".1f", cmap="Blues", cbar=True,
        xticklabels=class_names, yticklabels=class_names, ax=axes[0],
        vmin=0, vmax=100
    )
    mr_acc = accuracy_score(y_true, mr_preds) * 100
    axes[0].set_title(f"MiniRocket + Linear Classifier\nOverall Accuracy: {mr_acc:.2f}%", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Predicted Class", fontsize=11)
    axes[0].set_ylabel("True Class", fontsize=11)

    # CNN-LSTM Confusion Matrix
    sns.heatmap(
        cm_cnn, annot=True, fmt=".1f", cmap="Greens", cbar=True,
        xticklabels=class_names, yticklabels=class_names, ax=axes[1],
        vmin=0, vmax=100
    )
    cnn_acc = accuracy_score(y_true, cnn_preds) * 100
    axes[1].set_title(f"Hybrid CNN-LSTM Deep Learning\nOverall Accuracy: {cnn_acc:.2f}%", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Predicted Class", fontsize=11)
    axes[1].set_ylabel("True Class", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot saved] Confusion matrices saved to: {save_path}")


def plot_roc_curves(
    y_true: np.ndarray,
    mr_probs: np.ndarray,
    cnn_probs: np.ndarray,
    class_names: list,
    save_path: Path
):
    """Plot multi-class One-vs-Rest ROC curves for both models."""
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # MiniRocket ROC
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], mr_probs[:, i])
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f"{class_names[i]} (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.7)
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("MiniRocket ROC Curves (OvR)", fontsize=13, fontweight='bold')
    axes[0].legend(loc="lower right", frameon=True)

    # CNN-LSTM ROC
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], cnn_probs[:, i])
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                     label=f"{class_names[i]} (AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.7)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("False Positive Rate", fontsize=11)
    axes[1].set_ylabel("True Positive Rate", fontsize=11)
    axes[1].set_title("Hybrid CNN-LSTM ROC Curves (OvR)", fontsize=13, fontweight='bold')
    axes[1].legend(loc="lower right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Plot saved] ROC Curves saved to: {save_path}")


def plot_subject_benchmark_comparisons(df_summary: pd.DataFrame, plots_dir: Path):
    """Plot per-subject accuracy comparison and latency benchmark."""
    # 1. Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_summary))
    width = 0.35

    ax.bar(x - width/2, df_summary['minirocket_acc'] * 100, width, label='MiniRocket', color='#2b5c8f', edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, df_summary['cnn_lstm_acc'] * 100, width, label='CNN-LSTM', color='#2ca25f', edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Motor Imagery Accuracy by Subject: MiniRocket vs CNN-LSTM\n(Jamal Hwaidi & Mohamed Chahine Ghanem, NeuroImage 2026)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary['subject'], fontsize=11)
    ax.set_ylim([80, 103])
    ax.axhline(y=df_summary['minirocket_acc'].mean() * 100, color='#1d3557', linestyle='--', alpha=0.7, label=f'MiniRocket Mean ({df_summary["minirocket_acc"].mean()*100:.1f}%)')
    ax.axhline(y=df_summary['cnn_lstm_acc'].mean() * 100, color='#1b4d3e', linestyle=':', alpha=0.7, label=f'CNN-LSTM Mean ({df_summary["cnn_lstm_acc"].mean()*100:.1f}%)')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    acc_plot_path = plots_dir / "subject_accuracy_comparison.png"
    plt.savefig(acc_plot_path, dpi=300)
    plt.close()
    print(f"[Plot saved] Subject accuracy comparison saved to: {acc_plot_path}")

    # 2. Inference latency & speedup plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    latencies = [df_summary['minirocket_latency_ms'].mean(), df_summary['cnn_lstm_latency_ms'].mean()]
    models = ['MiniRocket', 'CNN-LSTM']
    colors = ['#386cb0', '#f0027f']

    bars = ax1.bar(models, latencies, color=colors, width=0.5, edgecolor='black')
    ax1.set_ylabel('Inference Latency per Trial (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Single-Trial Latency Comparison\n(Lower is Better)', fontsize=12, fontweight='bold')
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.15, f"{yval:.2f} ms", ha='center', va='bottom', fontweight='bold')

    # Speedup factor
    speedup = df_summary['speedup_factor'].mean()
    ax2.bar(['MiniRocket Speedup'], [speedup], color='#7fc97f', width=0.4, edgecolor='black')
    ax2.set_ylabel('Speedup Multiplier (x faster)', fontsize=11, fontweight='bold')
    ax2.set_title(f'Inference Acceleration Factor\nMiniRocket is {speedup:.1f}x Faster', fontsize=12, fontweight='bold')
    ax2.text(0, speedup / 2, f"{speedup:.1f}x\nFaster", ha='center', va='center', fontsize=16, fontweight='bold', color='white')

    plt.tight_layout()
    lat_plot_path = plots_dir / "inference_latency_speedup.png"
    plt.savefig(lat_plot_path, dpi=300)
    plt.close()
    print(f"[Plot saved] Latency speedup plot saved to: {lat_plot_path}")


def run_evaluation(
    dataset: str = "physionet",
    model_dir: str = "models",
    results_dir: str = "results"
):
    model_path = Path(model_dir)
    results_path = Path(results_dir)
    plots_dir = results_path / "plots"
    metrics_dir = results_path / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_file = metrics_dir / f"benchmark_summary_{dataset}.csv"
    if summary_file.exists():
        df_summary = pd.read_csv(summary_file)
        plot_subject_benchmark_comparisons(df_summary, plots_dir)

    test_data_file = model_path / f"sample_test_data_{dataset}.npz"
    mr_model_file = model_path / f"minirocket_{dataset}_s01.joblib"
    cnn_model_file = model_path / f"cnn_lstm_{dataset}_s01.pt"

    if test_data_file.exists() and mr_model_file.exists() and cnn_model_file.exists():
        print("\nLoading test split and fitted models for detailed evaluation...")
        data = np.load(test_data_file)
        X_test = data['X_test']
        y_test = data['y_test']

        class_names = [PHYSIONET_CLASSES[i] for i in sorted(PHYSIONET_CLASSES.keys())] if dataset == "physionet" else [BCI2A_CLASSES[i] for i in sorted(BCI2A_CLASSES.keys())]

        # MiniRocket predictions
        mr = MiniRocketPipeline.load(mr_model_file)
        mr_preds = mr.predict(X_test)
        mr_probs = mr.predict_proba(X_test)

        # CNN-LSTM predictions
        cnn = CNNLSTMPipeline.load(cnn_model_file, arch_type="primary")
        cnn_preds = cnn.predict(X_test)
        cnn_probs = cnn.predict_proba(X_test)

        # Plot Confusion Matrices
        plot_confusion_matrices(
            y_true=y_test,
            mr_preds=mr_preds,
            cnn_preds=cnn_preds,
            class_names=class_names,
            save_path=plots_dir / f"confusion_matrices_{dataset}.png"
        )

        # Plot ROC Curves
        plot_roc_curves(
            y_true=y_test,
            mr_probs=mr_probs,
            cnn_probs=cnn_probs,
            class_names=class_names,
            save_path=plots_dir / f"roc_curves_{dataset}.png"
        )

        # Save classification reports
        rep_mr = classification_report(y_test, mr_preds, target_names=class_names, output_dict=True)
        rep_cnn = classification_report(y_test, cnn_preds, target_names=class_names, output_dict=True)

        with open(metrics_dir / f"classification_report_minirocket_{dataset}.json", "w") as f:
            json.dump(rep_mr, f, indent=4)
        with open(metrics_dir / f"classification_report_cnnlstm_{dataset}.json", "w") as f:
            json.dump(rep_cnn, f, indent=4)
        print(f"[Metrics saved] Classification reports saved to: {metrics_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark plots and evaluation reports")
    parser.add_argument("--dataset", type=str, default="physionet", choices=["physionet", "bci_iv_2a"])
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    run_evaluation(dataset=args.dataset, model_dir=args.model_dir, results_dir=args.results_dir)
