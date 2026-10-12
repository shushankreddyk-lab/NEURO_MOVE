from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field


class DatasetReference(BaseModel):
    name: str
    dir_hint: str
    default_channels: int
    default_samples: int
    num_classes: int
    description: str = ""


class PreprocessingRequest(BaseModel):
    dataset: str
    subject_ids: list[int] = Field(default_factory=list)
    low_hz: float = 4.0
    high_hz: float = 38.0
    apply_car: bool = True


class PreprocessingStatus(BaseModel):
    ok: bool
    message: str
    dataset: str
    samples_processed: int = 0
    channels: int = 0
    samples_per_epoch: int = 0
    requested_low_hz: float = 4.0
    requested_high_hz: float = 38.0


class TrainJobRequest(BaseModel):
    dataset: str
    engine: str
    subject_ids: list[int] = Field(default_factory=list)
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 1e-3


class TrainJob(BaseModel):
    job_id: str
    status: str
    engine: str
    dataset: str
    progress: float = 0.0
    message: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class TrainJobUpdate(TrainJob):
    pass


class InferenceRequest(BaseModel):
    engine: str
    dataset: str
    sample_count: int = 32
    channels: int = 22
    samples: int = 656


class InferenceResult(BaseModel):
    ok: bool
    engine: str
    dataset: str
    num_samples: int
    classes: list[int]
    probabilities: list[list[float]] | None = None
    message: str = ""


class EvaluationRequest(BaseModel):
    dataset: str
    engine: str
    subject_ids: list[int] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    ok: bool
    dataset: str
    engine: str
    num_test_samples: int
    accuracy: float | None = None
    classes_found: list[int] = []
    message: str = ""


class HealthReport(BaseModel):
    ok: bool
    python: str
    engines: dict[str, object]
    venv_reachable: bool
    src_dir_exists: bool
    project_root_exists: bool
