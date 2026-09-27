"""
MiniRocket + RidgeClassifierCV Pipeline for EEG Motor Imagery Classification.
Reproducing Hwaidi & Ghanem (NeuroImage 328 (2026) 121816):
- 10,000 dilated convolutional kernels (or configurable for fast testing)
- PPV (Proportion of Positive Values) pooling
- RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
- Latency measurement (~0.6 ms per trial benchmark target)
"""

import time
import joblib
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Union
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.rocket import MiniRocketMultivariate


class MiniRocketPipeline:
    """
    MiniRocket transform paired with RidgeClassifierCV.
    """
    def __init__(
        self,
        num_kernels: int = 10000,
        max_dilations_per_kernel: int = 32,
        alphas: Optional[np.ndarray] = None,
        random_state: int = 42
    ):
        self.num_kernels = num_kernels
        self.max_dilations_per_kernel = max_dilations_per_kernel
        self.random_state = random_state
        
        if alphas is None:
            self.alphas = np.logspace(-3, 3, 10)
        else:
            self.alphas = alphas

        self.minirocket = MiniRocketMultivariate(
            num_kernels=self.num_kernels,
            max_dilations_per_kernel=self.max_dilations_per_kernel,
            random_state=self.random_state
        )
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.classifier = RidgeClassifierCV(alphas=self.alphas)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit MiniRocket kernels and train RidgeClassifierCV.
        
        Parameters
        ----------
        X : np.ndarray of shape (n_epochs, n_channels, n_times)
        y : np.ndarray of shape (n_epochs,)
        """
        t0 = time.perf_counter()
        # 1. Transform features via MiniRocket
        X_feat = self.minirocket.fit_transform(X)
        if hasattr(X_feat, 'to_numpy'):
            X_feat = X_feat.to_numpy()

        # 2. Scale features
        X_scaled = self.scaler.fit_transform(X_feat)

        # 3. Fit Ridge classifier
        self.classifier.fit(X_scaled, y)
        self.train_time_sec_ = time.perf_counter() - t0
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Extract MiniRocket features and scale.
        """
        X_feat = self.minirocket.transform(X)
        if hasattr(X_feat, 'to_numpy'):
            X_feat = X_feat.to_numpy()
        return self.scaler.transform(X_feat)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict motor imagery class labels.
        """
        X_scaled = self.transform(X)
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pseudo-probabilities using softmax over the decision function.
        """
        X_scaled = self.transform(X)
        decision = self.classifier.decision_function(X_scaled)
        if decision.ndim == 1:
            decision = np.vstack([-decision, decision]).T
        # Numerical stable softmax
        exp_d = np.exp(decision - np.max(decision, axis=-1, keepdims=True))
        return exp_d / np.sum(exp_d, axis=-1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate classification accuracy.
        """
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def benchmark_latency(self, single_sample: np.ndarray, n_warmup: int = 5, n_runs: int = 50) -> Dict[str, float]:
        """
        Benchmark single-trial inference latency in milliseconds.
        
        Parameters
        ----------
        single_sample : np.ndarray of shape (1, n_channels, n_times)
        """
        if single_sample.ndim == 2:
            single_sample = single_sample[np.newaxis, ...]

        # Warmup
        for _ in range(n_warmup):
            _ = self.predict(single_sample)

        # Benchmark
        timings = []
        for _ in range(n_runs):
            t_start = time.perf_counter()
            _ = self.predict(single_sample)
            t_end = time.perf_counter()
            timings.append((t_end - t_start) * 1000.0)  # ms

        timings = np.array(timings)
        return {
            "latency_mean_ms": float(np.mean(timings)),
            "latency_std_ms": float(np.std(timings)),
            "latency_p50_ms": float(np.median(timings)),
            "latency_p95_ms": float(np.percentile(timings, 95))
        }

    def save(self, filepath: Union[str, Path]):
        """Save fitted pipeline to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "MiniRocketPipeline":
        """Load fitted pipeline from disk."""
        return joblib.load(filepath)
