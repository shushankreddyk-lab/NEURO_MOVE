from __future__ import annotations

import logging
import sys
import time
import uuid
from typing import Any

import numpy as np

from neuromove.config import ARTIFACT_ROOT, PROJECT_ROOT, SRC_DIR, UPLOAD_ROOT
from neuromove.dependencies import import_engines
from neuromove.models import (
    EvaluationResult,
    InferenceResult,
    PreprocessingStatus,
    TrainJob,
    TrainJobUpdate,
)

logger = logging.getLogger(__name__)

_DATASET_INFO = {
    "PhysioNet EEGMMIDB": {
        "default_channels": 64,
        "default_samples": 656,
        "num_classes": 4,
        "dir_hint": str(PROJECT_ROOT / "physionet"),
        "description": "109-subject PhysioNet EEG Motor Movement/Imagery dataset.",
    },
    "BCI Competition IV 2a": {
        "default_channels": 22,
        "default_samples": 1025,
        "num_classes": 4,
        "dir_hint": str(PROJECT_ROOT / "BCICIV_2a_gdf"),
        "description": "BCI Competition IV 2a, 9 subjects, 4 MI classes.",
    },
    "High-Gamma Dataset": {
        "default_channels": 128,
        "default_samples": 2000,
        "num_classes": 4,
        "dir_hint": str(PROJECT_ROOT / "data" / "NEMAR" / "nm000172"),
        "description": "High-Gamma Dataset, motor execution.",
    },
    "Kaya Finger Movements": {
        "default_channels": 32,
        "default_samples": 4000,
        "num_classes": 4,
        "dir_hint": str(PROJECT_ROOT / "data" / "Kaya_Finger_Movements"),
        "description": "Kaya Finger Movements dataset.",
    },
    "WAY-EEG-GAL": {
        "default_channels": 32,
        "default_samples": 2000,
        "num_classes": 6,
        "dir_hint": str(PROJECT_ROOT / "dataset" / "grasp-and-lift-eeg-detection"),
        "description": "WAY-EEG-GAL grasp and lift detection.",
    },
    "NEMAR Finger MI": {
        "default_channels": 65,
        "default_samples": 4000,
        "num_classes": 5,
        "dir_hint": str(PROJECT_ROOT / "dataset" / "NEMAR_on008446"),
        "description": "NEMAR Finger Motor Imagery dataset.",
    },
}

ENGINE_ALIASES = {
    "minirocket": "MiniRocketPipeline",
    "conformer": "AdvancedEEGPipeline",
    "cnn_lstm": "CNN_LSTM_Pipeline",
    "eegnet": "EEGNet_Pipeline",
    "convnets": "ConvNet_Pipeline",
}

# In-memory job store for the demo backend.
_jobs: dict[str, TrainJob] = {}


def _engine_instance(engines: dict[str, Any], name: str, **kwargs: Any) -> Any:
    module = engines.get(name)
    if isinstance(module, dict) and "error" in module:
        raise RuntimeError(f"Engine {name} failed to import: {module['error']}")
    if not hasattr(module, ENGINE_ALIASES.get(name)):
        raise RuntimeError(f"Engine module {name} does not expose {ENGINE_ALIASES[name]}")
    cls = getattr(module, ENGINE_ALIASES[name])
    return cls(**kwargs)


class NeuroMoveService:
    def __init__(self) -> None:
        self.engines = import_engines()
        self._preproc_cache: dict[str, PreprocessingStatus] = {}

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "python": sys_version(),
            "engines": summarize_engines(self.engines),
            "venv_reachable": PROJECT_ROOT.joinpath(".venv", "Scripts", "python.exe").exists(),
            "src_dir_exists": SRC_DIR.exists(),
            "project_root_exists": PROJECT_ROOT.exists(),
        }

    # ------------------------------------------------------------------
    # Datasets
    # ------------------------------------------------------------------

    def dataset_list(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, info in _DATASET_INFO.items():
            rows.append({
                "name": name,
                "default_channels": info["default_channels"],
                "default_samples": info["default_samples"],
                "num_classes": info["num_classes"],
                "dir_hint": info["dir_hint"],
                "description": info["description"],
                "directory_exists": Path(info["dir_hint"]).exists(),
            })
        return rows

    # ------------------------------------------------------------------
    # Preprocessing placeholder
    # ------------------------------------------------------------------

    def preprocess(self, dataset: str, low_hz: float = 4.0, high_hz: float = 38.0, apply_car: bool = True) -> PreprocessingStatus:
        info = _DATASET_INFO.get(dataset)
        if info is None:
            return PreprocessingStatus(
                ok=False,
                message=f"Unknown dataset: {dataset}",
                dataset=dataset,
            )

        # Real preprocessing requires actual data on disk. For the release backend
        # we report the requested configuration honestly rather than invent results.
        status = PreprocessingStatus(
            ok=True,
            message="Preprocessing configuration received. Actual processing requires dataset files on disk.",
            dataset=dataset,
            samples_processed=0,
            channels=info["default_channels"],
            samples_per_epoch=info["default_samples"],
            requested_low_hz=low_hz,
            requested_high_hz=high_hz,
        )
        self._preproc_cache[dataset] = status
        return status

    # ------------------------------------------------------------------
    # Training jobs
    # ------------------------------------------------------------------

    def submit_training(self, dataset: str, engine: str, epochs: int = 10, batch_size: int = 256, learning_rate: float = 1e-3) -> TrainJob:
        job_id = uuid.uuid4().hex[:12]
        job = TrainJob(
            job_id=job_id,
            status="pending",
            engine=engine,
            dataset=dataset,
            message=f"Training job queued for {engine} on {dataset}.",
        )
        _jobs[job_id] = job
        logger.info("queued training job %s: %s on %s", job_id, engine, dataset)
        return job

    def job_status(self, job_id: str) -> TrainJobUpdate:
        job = _jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job id: {job_id}")
        return TrainJobUpdate.model_validate(job.model_dump())

    def run_small_training_preview(self, job_id: str, engine: str, dataset: str, epochs: int, batch_size: int, learning_rate: float = 1e-3) -> TrainJobUpdate:
        """Run a tiny synthetic training preview so the UI can see a real completion
        flow without requiring full datasets on disk for every demo.

        This is intentionally a small synthetic sanity check. Real training uses
        the existing engines on real data.
        """
        job = _jobs.get(job_id)
        if job is None:
            raise KeyError(f"Unknown job id: {job_id}")

        try:
            engines = import_engines()
            if engine == "minirocket":
                pipeline = _engine_instance(
                    engines,
                    "minirocket",
                    num_kernels=1000,
                    in_channels=22,
                    seq_len=656,
                    head_epochs=max(1, epochs),
                    head_lr=learning_rate,
                    batch_size=batch_size,
                )
                X = np.random.randn(64, 22, 656).astype(np.float32)
                y = np.array([0, 1, 2, 3] * 16, dtype=int)
                pipeline.fit(X, y)
                acc = 0.0
                if hasattr(pipeline, "predict"):
                    pred = pipeline.predict(X[:8])
                    acc = float(np.mean(pred == y[:8]))
                job.status = "completed"
                job.progress = 1.0
                job.message = f"Synthetic MiniRocket preview finished. sanity accuracy={acc:.3f}"
            elif engine == "conformer":
                pipeline = _engine_instance(
                    engines,
                    "conformer",
                    num_classes=4,
                    channels=22,
                    samples=656,
                    epochs=max(1, epochs),
                    batch_size=batch_size,
                    lr=learning_rate,
                )
                X = np.random.randn(64, 22, 656).astype(np.float32)
                y = np.array([0, 1, 2, 3] * 16, dtype=int)
                lut = {c: i for i, c in enumerate(np.unique(y))}
                yi = np.array([lut[v] for v in y], dtype=int)
                pipeline.fit(X, yi)
                job.status = "completed"
                job.progress = 1.0
                job.message = "Synthetic Conformer preview finished."
            else:
                job.status = "completed"
                job.progress = 1.0
                job.message = f"Engine placeholder completed for {engine}."
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.message = repr(exc)
        finally:
            _jobs[job_id] = job
        return TrainJobUpdate.model_validate(job.model_dump())

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, engine: str, dataset: str, sample_count: int = 32, channels: int = 22, samples: int = 656) -> InferenceResult:
        try:
            engines = import_engines()
            if engine == "minirocket":
                pipeline = _engine_instance(
                    engines,
                    "minirocket",
                    num_kernels=1000,
                    in_channels=channels,
                    seq_len=samples,
                    head_epochs=1,
                    head_lr=1e-3,
                    batch_size=64,
                )
                X = np.random.randn(sample_count, channels, samples).astype(np.float32)
                y = np.arange(sample_count) % 4
                pipeline.fit(X, y)
                proba = pipeline.predict_proba(X)
                num_classes = int(proba.shape[1])
                preds = np.argmax(proba, axis=1).tolist()
                return InferenceResult(
                    ok=True,
                    engine=engine,
                    dataset=dataset,
                    num_samples=sample_count,
                    classes=list(range(num_classes)),
                    probabilities=proba.tolist(),
                    message="Inference completed on synthetic input.",
                )
            elif engine == "conformer":
                pipeline = _engine_instance(
                    engines,
                    "conformer",
                    num_classes=4,
                    channels=channels,
                    samples=samples,
                    epochs=1,
                    batch_size=64,
                    lr=1e-3,
                )
                X = np.random.randn(sample_count, channels, samples).astype(np.float32)
                y = np.arange(sample_count) % 4
                lut = {c: i for i, c in enumerate(np.unique(y))}
                yi = np.array([lut[v] for v in y], dtype=int)
                pipeline.fit(X, yi)
                proba = pipeline.predict_proba(X)
                num_classes = int(proba.shape[1])
                preds = np.argmax(proba, axis=1).tolist()
                return InferenceResult(
                    ok=True,
                    engine=engine,
                    dataset=dataset,
                    num_samples=sample_count,
                    classes=list(range(num_classes)),
                    probabilities=proba.tolist(),
                    message="Inference completed on synthetic input.",
                )
            else:
                return InferenceResult(
                    ok=False,
                    engine=engine,
                    dataset=dataset,
                    num_samples=sample_count,
                    classes=[],
                    message=f"Engine {engine} is not wired for inference yet.",
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("inference failed")
            return InferenceResult(
                ok=False,
                engine=engine,
                dataset=dataset,
                num_samples=sample_count,
                classes=[],
                message=repr(exc),
            )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, dataset: str, engine: str) -> EvaluationResult:
        try:
            engines = import_engines()
            if engine not in engines or isinstance(engines[engine], dict):
                return EvaluationResult(
                    ok=False,
                    dataset=dataset,
                    engine=engine,
                    num_test_samples=0,
                    accuracy=None,
                    message=f"Engine {engine} is not available.",
                )
            return EvaluationResult(
                ok=True,
                dataset=dataset,
                engine=engine,
                num_test_samples=0,
                accuracy=None,
                message="Real evaluation requires dataset files on disk and a trained model. This endpoint is wired but not emitting fabricated metrics.",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("evaluation failed")
            return EvaluationResult(
                ok=False,
                dataset=dataset,
                engine=engine,
                num_test_samples=0,
                accuracy=None,
                message=repr(exc),
            )


def sys_version() -> str:
    import sys

    return f"{sys.version.splitlines()[0].strip()}"


def summarize_engines(engines: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, mod in engines.items():
        if isinstance(mod, dict) and "error" in mod:
            out[name] = {"available": False, "error": mod["error"]}
        elif hasattr(mod, "__name__"):
            out[name] = {"available": True, "module": mod.__name__}
        else:
            out[name] = {"available": False, "note": "unusual import result"}
    return out


from pathlib import Path  # noqa: E402
