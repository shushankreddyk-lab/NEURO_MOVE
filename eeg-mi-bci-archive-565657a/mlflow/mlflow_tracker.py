"""
MLflow Integration for NeuroMove Model Training Tracking.
"""
import os
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
from datetime import datetime


class NeuroMoveMLflowTracker:
    """MLflow tracking wrapper for NeuroMove experiments."""
    
    def __init__(
        self,
        experiment_name: str = "neuromove-training",
        tracking_uri: str = None,
        registry_uri: str = None
    ):
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("file:./mlruns")
        
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        # Create or get experiment
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        if not self.experiment:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = self.experiment.experiment_id
        
        self.client = MlflowClient()
        self.active_run = None
    
    def start_run(
        self,
        run_name: str = None,
        tags: Dict[str, str] = None,
        architecture: str = "",
        dataset: str = "",
        subjects: str = ""
    ) -> mlflow.ActiveRun:
        """Start a new MLflow run with standard tags."""
        tags = tags or {}
        tags.update({
            "project": "neuromove",
            "architecture": architecture,
            "dataset": dataset,
            "subjects": subjects,
            "created_by": "neuromove-pipeline"
        })
        
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name,
            tags=tags
        )
        return self.active_run
    
    def log_config(self, config: Dict[str, Any], prefix: str = "") -> None:
        """Log configuration parameters."""
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                mlflow.log_param(f"{prefix}{key}", json.dumps(value))
            else:
                mlflow.log_param(f"{prefix}{key}", value)
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None) -> None:
        """Log metrics."""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(key, value, step=step)
    
    def log_model_artifacts(
        self,
        model,
        model_name: str,
        model_path: str,
        architecture: str,
        dataset: str,
        subjects: str,
        accuracy: float,
        extra_files: List[str] = None
    ) -> str:
        """Log model and register in MLflow Model Registry."""
        # Log model
        if architecture == "minirocket":
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"neuromove-{architecture}-{dataset}"
            )
        else:
            mlflow.pytorch.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"neuromove-{architecture}-{dataset}"
            )
        
        # Log additional artifacts
        if extra_files:
            for f in extra_files:
                mlflow.log_artifact(f)
        
        # Register model version
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mv = mlflow.register_model(
            model_uri,
            f"neuromove-{architecture}-{dataset}",
            tags={
                "architecture": architecture,
                "dataset": dataset,
                "subjects": subjects,
                "accuracy": str(accuracy),
                "model_name": model_name
            }
        )
        
        # Set aliases
        self.client.set_registered_model_alias(
            f"neuromove-{architecture}-{dataset}",
            "champion",
            mv.version
        )
        self.client.set_registered_model_alias(
            f"neuromove-{architecture}-{dataset}",
            f"v{mv.version}",
            mv.version
        )
        
        return mv.version
    
    def log_dataset_info(
        self,
        dataset_name: str,
        n_subjects: int,
        n_channels: int,
        n_samples: int,
        sfreq: float,
        n_trials: int,
        class_distribution: Dict[str, int]
    ) -> None:
        """Log dataset metadata."""
        mlflow.log_params({
            "dataset_name": dataset_name,
            "n_subjects": n_subjects,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "sfreq": sfreq,
            "n_trials": n_trials,
        })
        mlflow.log_param("class_distribution", json.dumps(class_distribution))
    
    def log_training_history(self, history: Dict[str, List[float]]) -> None:
        """Log training curves as artifacts."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='Train')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        if 'train_acc' in history:
            axes[0, 1].plot(history['train_acc'], label='Train')
        if 'val_acc' in history:
            axes[0, 1].plot(history['val_acc'], label='Val')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate
        if 'lr' in history:
            axes[1, 0].plot(history['lr'])
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].grid(True)
        
        # F1
        if 'train_f1' in history:
            axes[1, 1].plot(history['train_f1'], label='Train')
        if 'val_f1' in history:
            axes[1, 1].plot(history['val_f1'], label='Val')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=150)
        plt.close()
        mlflow.log_artifact('training_curves.png')
        os.remove('training_curves.png')
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        mlflow.end_run(status=status)
        self.active_run = None


def get_best_model(
    architecture: str,
    dataset: str,
    metric: str = "accuracy",
    ascending: bool = False
) -> Optional[Dict[str, Any]]:
    """Get best model from MLflow Model Registry."""
    client = MlflowClient()
    model_name = f"neuromove-{architecture}-{dataset}"
    
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            return None
        
        # Get run metrics for each version
        best_version = None
        best_metric = -float('inf') if not ascending else float('inf')
        
        for v in versions:
            run = client.get_run(v.run_id)
            if metric in run.data.metrics:
                value = run.data.metrics[metric]
                if (not ascending and value > best_metric) or (ascending and value < best_metric):
                    best_metric = value
                    best_version = v
        
        if best_version:
            return {
                "version": best_version.version,
                "run_id": best_version.run_id,
                "model_uri": f"models:/{model_name}/{best_version.version}",
                "metric_value": best_metric,
                "metric_name": metric
            }
        return None
    except Exception as e:
        print(f"Error getting best model: {e}")
        return None


def load_model_from_registry(
    architecture: str,
    dataset: str,
    version: str = None
) -> Any:
    """Load model from MLflow Model Registry."""
    model_name = f"neuromove-{architecture}-{dataset}"
    
    if version:
        model_uri = f"models:/{model_name}/{version}"
    else:
        # Use champion alias
        model_uri = f"models:/{model_name}@champion"
    
    if architecture == "minirocket":
        return mlflow.sklearn.load_model(model_uri)
    else:
        return mlflow.pytorch.load_model(model_uri)


def compare_runs(
    experiment_name: str = "neuromove-training",
    filter_string: str = "",
    order_by: List[str] = None
) -> List[Dict[str, Any]]:
    """Compare runs in an experiment."""
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        return []
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=order_by or ["metrics.accuracy DESC"],
        max_results=100,
        run_view_type=ViewType.ACTIVE_ONLY
    )
    
    results = []
    for run in runs:
        results.append({
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "architecture": run.data.tags.get("architecture", ""),
            "dataset": run.data.tags.get("dataset", ""),
            "subjects": run.data.tags.get("subjects", ""),
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags
        })
    return results


# Context manager for automatic run management
class TrainingRun:
    """Context manager for MLflow training runs."""
    
    def __init__(
        self,
        tracker: NeuroMoveMLflowTracker,
        run_name: str,
        architecture: str,
        dataset: str,
        subjects: str,
        config: Dict[str, Any] = None
    ):
        self.tracker = tracker
        self.run_name = run_name
        self.architecture = architecture
        self.dataset = dataset
        self.subjects = subjects
        self.config = config or {}
        self.run = None
    
    def __enter__(self):
        self.run = self.tracker.start_run(
            run_name=self.run_name,
            architecture=self.architecture,
            dataset=self.dataset,
            subjects=self.subjects
        )
        if self.config:
            self.tracker.log_config(self.config)
        return self
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        self.tracker.log_metrics(metrics, step)
    
    def log_params(self, params: Dict[str, Any]):
        self.tracker.log_config(params)
    
    def log_artifact(self, path: str):
        mlflow.log_artifact(path)
    
    def log_model(self, model, model_name: str, accuracy: float, extra_files: List[str] = None):
        return self.tracker.log_model_artifacts(
            model, model_name, "", self.architecture, self.dataset,
            self.subjects, accuracy, extra_files
        )
    
    def log_dataset(self, **kwargs):
        self.tracker.log_dataset_info(**kwargs)
    
    def log_training_history(self, history: Dict[str, List[float]]):
        self.tracker.log_training_history(history)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        self.tracker.end_run(status)
        return False


if __name__ == "__main__":
    # Example usage
    tracker = NeuroMoveMLflowTracker()
    
    with TrainingRun(
        tracker,
        run_name="minirocket_physionet_subs1-10",
        architecture="minirocket",
        dataset="physionet",
        subjects="1-10",
        config={"num_kernels": 10000, "head_epochs": 30, "batch_size": 256}
    ) as run:
        # Simulate training
        run.log_params({"lr": 5e-4, "weight_decay": 5e-3})
        run.log_dataset(
            dataset_name="physionet",
            n_subjects=10,
            n_channels=20,
            n_samples=656,
            sfreq=160,
            n_trials=400,
            class_distribution={"0": 100, "1": 100, "2": 100, "3": 100}
        )
        
        # Simulate epochs
        for epoch in range(10):
            run.log_metrics({
                "train_loss": 0.5 - epoch * 0.03,
                "val_loss": 0.6 - epoch * 0.02,
                "train_acc": 0.7 + epoch * 0.02,
                "val_acc": 0.68 + epoch * 0.02,
                "lr": 5e-4 * (0.95 ** epoch)
            }, step=epoch)
        
        run.log_training_history({
            "train_loss": [0.5, 0.47, 0.44, 0.41, 0.38, 0.35, 0.32, 0.29, 0.26, 0.23],
            "val_loss": [0.6, 0.58, 0.56, 0.54, 0.52, 0.50, 0.48, 0.46, 0.44, 0.42],
            "train_acc": [0.7, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88],
            "val_acc": [0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86],
            "lr": [5e-4 * (0.95**i) for i in range(10)]
        })
        
        # Log final metrics
        run.log_metrics({
            "final_train_acc": 0.88,
            "final_val_acc": 0.86,
            "final_train_loss": 0.23,
            "final_val_loss": 0.42
        })
        
        # Model would be logged here
        # run.log_model(pipeline, "minirocket_physionet_subs1-10", 0.86)