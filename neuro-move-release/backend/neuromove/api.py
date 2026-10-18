from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from neuromove.models import (
    EvaluationRequest,
    EvaluationResult,
    HealthReport,
    InferenceRequest,
    InferenceResult,
    PreprocessingStatus,
    TrainJob,
    TrainJobRequest,
    TrainJobUpdate,
)
from neuromove.service import NeuroMoveService
from neuromove.config import ensure_paths

ensure_paths()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("neuromove")

app = FastAPI(
    title="NeuroMove Backend",
    version="0.1.0",
    description="Backend for the NeuroMove MI-BCI release.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

service = NeuroMoveService()


@app.get("/health", response_model=HealthReport)
def health() -> HealthReport:
    return HealthReport(**service.health())


@app.get("/datasets")
def dataset_list() -> list[dict[str, object]]:
    return service.dataset_list()


@app.post("/preprocess", response_model=PreprocessingStatus)
def preprocess(body: dict) -> PreprocessingStatus:
    dataset = body.get("dataset", "")
    low_hz = float(body.get("low_hz", 4.0))
    high_hz = float(body.get("high_hz", 38.0))
    apply_car = bool(body.get("apply_car", True))
    return service.preprocess(dataset, low_hz=low_hz, high_hz=high_hz, apply_car=apply_car)


@app.post("/train", response_model=TrainJob)
def submit_training(body: TrainJobRequest) -> TrainJob:
    return service.submit_training(
        dataset=body.dataset,
        engine=body.engine,
        epochs=body.epochs,
        batch_size=body.batch_size,
        learning_rate=body.learning_rate,
    )


@app.get("/train/{job_id}", response_model=TrainJobUpdate)
def job_status(job_id: str) -> TrainJobUpdate:
    try:
        return service.job_status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/train/{job_id}/run-preview", response_model=TrainJobUpdate)
def run_preview(job_id: str, body: TrainJobRequest) -> TrainJobUpdate:
    return service.run_small_training_preview(
        job_id=job_id,
        engine=body.engine,
        dataset=body.dataset,
        epochs=body.epochs,
        batch_size=body.batch_size,
        learning_rate=body.learning_rate,
    )


@app.post("/infer", response_model=InferenceResult)
def infer(body: InferenceRequest) -> InferenceResult:
    return service.infer(
        engine=body.engine,
        dataset=body.dataset,
        sample_count=body.sample_count,
        channels=body.channels,
        samples=body.samples,
    )


@app.post("/evaluate", response_model=EvaluationResult)
def evaluate(body: EvaluationRequest) -> EvaluationResult:
    return service.evaluate(dataset=body.dataset, engine=body.engine)
