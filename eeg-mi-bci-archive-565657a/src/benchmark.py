"""
Benchmark Runner — Compare all architectures on standard splits.
"""
import time
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class BenchmarkResult:
    architecture: str
    dataset: str
    subjects: str
    n_trials_train: int
    n_trials_test: int
    accuracy: float
    f1_macro: float
    f1_weighted: float
    training_time_s: float
    inference_latency_ms: float
    model_size_mb: float
    n_parameters: int
    config: Dict[str, Any]
    timestamp: str


class BenchmarkRunner:
    """Run standardized benchmarks across architectures and datasets."""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
    
    def _get_architecture_configs(self) -> Dict[str, Dict]:
        """Return default configs for each architecture."""
        return {
            'minirocket': {
                'class': 'MiniRocketPipeline',
                'module': 'minirocket_engine',
                'kwargs': {
                    'num_kernels': 10000,
                    'hidden': 512,
                    'dropout': 0.15,
                    'head_epochs': 30,
                    'head_lr': 5e-4,
                    'batch_size': 256,
                },
            },
            'eegnet': {
                'class': 'EEGNet_Pipeline',
                'module': 'eegnet_engine',
                'kwargs': {
                    'num_classes': 4,
                    'channels': 20,
                    'samples': 656,
                    'dropout': 0.5,
                    'kernel_length': 64,
                },
            },
            'cnn_lstm': {
                'class': 'CNN_LSTM_Pipeline',
                'module': 'cnn_lstm_engine',
                'kwargs': {
                    'num_classes': 4,
                    'channels': 20,
                    'samples': 656,
                    'dropout': 0.3,
                },
            },
            'shallow_convnet': {
                'class': 'ConvNet_Pipeline',
                'module': 'convnets_engine',
                'kwargs': {
                    'arch': 'shallow',
                    'num_classes': 4,
                    'channels': 20,
                    'samples': 656,
                },
            },
            'deep_convnet': {
                'class': 'ConvNet_Pipeline',
                'module': 'convnets_engine',
                'kwargs': {
                    'arch': 'deep',
                    'num_classes': 4,
                    'channels': 20,
                    'samples': 656,
                },
            },
            'advanced_eeg': {
                'class': 'AdvancedEEGPipeline',
                'module': 'advanced_eeg_engine',
                'kwargs': {
                    'num_classes': 4,
                    'channels': 20,
                    'samples': 656,
                    'model_type': 'conformer',
                },
            },
        }
    
    def _import_pipeline(self, module_name: str, class_name: str):
        """Dynamically import pipeline class."""
        import importlib
        module = importlib.import_module(f'src.{module_name}')
        return getattr(module, class_name)
    
    def run_single(
        self,
        architecture: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset: str,
        subjects: str,
        custom_kwargs: Dict = None,
    ) -> BenchmarkResult:
        """Run single architecture benchmark."""
        arch_configs = self._get_architecture_configs()
        
        if architecture not in arch_configs:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        config = arch_configs[architecture]
        if custom_kwargs:
            config['kwargs'].update(custom_kwargs)
        
        PipelineClass = self._import_pipeline(config['module'], config['class'])
        
        # Initialize
        pipeline = PipelineClass(**config['kwargs'])
        
        # Train
        start_train = time.time()
        pipeline.fit(X_train, y_train)
        training_time = time.time() - start_train
        
        # Test
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        from evaluation import compute_metrics
        metrics = compute_metrics(y_test, y_pred)
        
        # Inference latency (average over 10 runs)
        latencies = []
        for _ in range(10):
            start = time.time()
            _ = pipeline.predict(X_test[:1])
            latencies.append((time.time() - start) * 1000)
        avg_latency = np.mean(latencies)
        
        # Model size
        import torch
        n_params = sum(p.numel() for p in pipeline.parameters() if hasattr(p, 'parameters'))
        # Save temp to get file size
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            temp_path = f.name
        try:
            pipeline.save(temp_path)
            model_size_mb = Path(temp_path).stat().st_size / 1024 / 1024
            Path(temp_path).unlink()
        except Exception:
            model_size_mb = 0.0
        
        result = BenchmarkResult(
            architecture=architecture,
            dataset=dataset,
            subjects=subjects,
            n_trials_train=len(X_train),
            n_trials_test=len(X_test),
            accuracy=metrics['accuracy'],
            f1_macro=metrics['f1_macro'],
            f1_weighted=metrics['f1_weighted'],
            training_time_s=training_time,
            inference_latency_ms=avg_latency,
            model_size_mb=model_size_mb,
            n_parameters=n_params,
            config=config['kwargs'],
            timestamp=datetime.utcnow().isoformat() + 'Z',
        )
        
        self.results.append(result)
        return result
    
    def run_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dataset: str,
        subjects: str,
        architectures: List[str] = None,
    ) -> List[BenchmarkResult]:
        """Run benchmark on all architectures."""
        if architectures is None:
            architectures = list(self._get_architecture_configs().keys())
        
        for arch in architectures:
            print(f"\n{'='*60}")
            print(f"Benchmarking {arch} on {dataset} ({subjects})")
            print(f"{'='*60}")
            
            try:
                result = self.run_single(arch, X_train, y_train, X_test, y_test, dataset, subjects)
                print(f"✓ {arch}: Acc={result.accuracy*100:.2f}%, "
                      f"Latency={result.inference_latency_ms:.1f}ms, "
                      f"Train={result.training_time_s:.1f}s")
            except Exception as e:
                print(f"✗ {arch}: Failed - {e}")
                import traceback
                traceback.print_exc()
        
        return self.results
    
    def save_results(self, filename: str = None):
        """Save all results to JSON."""
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = self.output_dir / filename
        data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'results': [asdict(r) for r in self.results],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nSaved {len(self.results)} results to {filepath}")
        return filepath
    
    def print_summary(self):
        """Print benchmark summary table."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "="*100)
        print(f"{'Architecture':<20} {'Dataset':<15} {'Subjects':<12} {'Acc%':>8} {'F1%':>8} {'Train(s)':>10} {'Latency(ms)':>12} {'Size(MB)':>10}")
        print("-"*100)
        
        for r in self.results:
            print(f"{r.architecture:<20} {r.dataset:<15} {r.subjects:<12} "
                  f"{r.accuracy*100:>7.2f}% {r.f1_macro*100:>7.2f}% "
                  f"{r.training_time_s:>9.1f}s {r.inference_latency_ms:>11.1f}ms "
                  f"{r.model_size_mb:>9.2f}MB")
        
        print("="*100)
        
        # Best by accuracy
        best = max(self.results, key=lambda r: r.accuracy)
        print(f"\nBest Accuracy: {best.architecture} ({best.accuracy*100:.2f}%)")
        
        # Fastest inference
        fastest = min(self.results, key=lambda r: r.inference_latency_ms)
        print(f"Fastest Inference: {fastest.architecture} ({fastest.inference_latency_ms:.1f}ms)")
        
        # Smallest model
        smallest = min(self.results, key=lambda r: r.model_size_mb)
        print(f"Smallest Model: {smallest.architecture} ({smallest.model_size_mb:.2f}MB)")


def run_benchmark_suite(
    output_dir: str = "benchmark_results",
    dataset: str = "physionet",
    subjects: str = "1-109",
    architectures: List[str] = None,
) -> List[BenchmarkResult]:
    """Convenience function to run full benchmark suite."""
    # This would load real data - placeholder uses synthetic
    from scripts.generate_synthetic import generate_synthetic_eeg
    from sklearn.model_selection import train_test_split
    
    print(f"Generating synthetic data for {dataset}...")
    X, y, _ = generate_synthetic_eeg(
        n_trials=500,
        dataset=dataset,
        n_channels=20 if dataset != 'bci2a' else 22,
        n_samples=656 if dataset != 'bci2a' else 656,
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    runner = BenchmarkRunner(output_dir)
    results = runner.run_all(X_train, y_train, X_test, y_test, dataset, subjects, architectures)
    runner.print_summary()
    runner.save_results()
    
    return results


if __name__ == "__main__":
    # Quick demo with synthetic data
    run_benchmark_suite(
        architectures=['minirocket', 'eegnet'],  # Subset for demo
        dataset='physionet',
        subjects='1-10',
    )