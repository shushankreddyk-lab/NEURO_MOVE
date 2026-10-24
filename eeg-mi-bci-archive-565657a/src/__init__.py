"""
NeuroMove Enterprise MI-BCI — Core Engines Package.

Canonical implementations for Motor Imagery BCI decoding.
"""
from pathlib import Path
import sys

# Ensure archive src is in path for unified imports
_archive_src = Path(__file__).resolve().parent
if str(_archive_src) not in sys.path:
    sys.path.insert(0, str(_archive_src))

# Core pipelines
from minirocket_engine import (
    MiniRocketPipeline,
    TorchMiniRocket,
    GPUMiniRocketHead,
    get_device,
)

from eegnet_engine import EEGNet_Pipeline
from cnn_lstm_engine import CNN_LSTM_Pipeline
from convnets_engine import ConvNet_Pipeline
from advanced_eeg_engine import AdvancedEEGPipeline
from transformer_engine import TransformerEngine

# Data & preprocessing
from preprocessing import (
    apply_car,
    apply_bandpass_filter,
    spatial_channel_augmentation,
    epoch_and_segment,
    zscore_normalize,
)

from binary_parser import (
    load_local_eeg_data,
    normalize_channel_names,
    parse_edf_events,
    parse_gdf_events,
)

from dataset_loader_all import load_dataset
from dataset_2a_loader import load_bci_2a_data

# Training & evaluation
from train_master import (
    extract_and_save_data,
    train_model_group,
    map_run_and_marker_to_group,
)

from evaluation import (
    compute_metrics,
    confusion_matrix_plot,
    classification_report_str,
    loso_cv_scores,
    evaluate_model_on_subjects,
    save_evaluation_results,
    load_evaluation_results,
    print_loso_summary,
)

# Benchmark
from benchmark import (
    BenchmarkRunner,
    BenchmarkResult,
    run_benchmark_suite,
)

# Utilities
from config_loader import get_config, Config
from model_registry import ModelRegistry, get_registry
from visualizer import (
    plot_eeg_trial,
    plot_topomap,
    plot_confusion_matrix,
    plot_training_curves,
)

from engines_safe_print import safe_print
from realtime_inference_sim import RealTimeInferenceSimulator
from exploratory_analysis import run_exploratory_analysis

# Synthetic data
from scripts.generate_synthetic import (
    generate_synthetic_eeg,
    generate_continuous_eeg,
    generate_mu_rhythm,
    generate_erd_pattern,
    save_synthetic_dataset,
)

__version__ = "1.0.0"

__all__ = [
    # Pipelines
    "MiniRocketPipeline",
    "TorchMiniRocket",
    "GPUMiniRocketHead",
    "get_device",
    "EEGNet_Pipeline",
    "CNN_LSTM_Pipeline",
    "ConvNet_Pipeline",
    "AdvancedEEGPipeline",
    "TransformerEngine",
    
    # Preprocessing
    "apply_car",
    "apply_bandpass_filter",
    "spatial_channel_augmentation",
    "epoch_and_segment",
    "zscore_normalize",
    
    # Data loading
    "load_local_eeg_data",
    "normalize_channel_names",
    "parse_edf_events",
    "parse_gdf_events",
    "load_dataset",
    "load_bci_2a_data",
    
    # Training
    "extract_and_save_data",
    "train_model_group",
    "map_run_and_marker_to_group",
    
    # Evaluation
    "compute_metrics",
    "confusion_matrix_plot",
    "classification_report_str",
    "loso_cv_scores",
    "evaluate_model_on_subjects",
    "save_evaluation_results",
    "load_evaluation_results",
    "print_loso_summary",
    
    # Benchmark
    "BenchmarkRunner",
    "BenchmarkResult",
    "run_benchmark_suite",
    
    # Config & Registry
    "get_config",
    "Config",
    "ModelRegistry",
    "get_registry",
    
    # Visualization
    "plot_eeg_trial",
    "plot_topomap",
    "plot_confusion_matrix",
    "plot_training_curves",
    
    # Real-time
    "RealTimeInferenceSimulator",
    "run_exploratory_analysis",
    
    # Synthetic data
    "generate_synthetic_eeg",
    "generate_continuous_eeg",
    "generate_mu_rhythm",
    "generate_erd_pattern",
    "save_synthetic_dataset",
    
    # Utils
    "safe_print",
]

# Convenience function
def get_available_architectures() -> list:
    """Return list of available model architectures."""
    return [
        "minirocket",
        "eegnet",
        "cnn_lstm",
        "shallow_convnet",
        "deep_convnet",
        "advanced_eeg",
        "transformer",
    ]


def get_supported_datasets() -> list:
    """Return list of supported dataset names."""
    return [
        "physionet_eegmmidb",
        "bci_competition_iv_2a",
        "high_gamma",
        "kaya_fingers",
        "way_eeg_gal",
        "nemar_finger_mi",
    ]


if __name__ == "__main__":
    print(f"NeuroMove Archive v{__version__}")
    print(f"Architectures: {get_available_architectures()}")
    print(f"Datasets: {get_supported_datasets()}")