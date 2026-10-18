"""
Inference endpoints
"""
from fastapi import APIRouter, Depends, HTTPException
from app.core.security import get_current_user, require_clinician, verify_turnstile_dependency
from app.db.database import AsyncSessionLocal
from app.models.base import InferenceJob, Model, User
from app.schemas.base import InferenceRequest, InferenceResponse
from app.core.config import settings
import numpy as np, time
from sqlalchemy import select
import uuid

router = APIRouter()


@router.post("", response_model=InferenceResponse)
async def inference(
    request: InferenceRequest,
    turnstile_verified: bool = Depends(verify_turnstile_dependency),
    current_user: dict = Depends(require_clinician),
):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Model).where(Model.name == request.model_name))
        model = result.scalar_one_or_none()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

    start_time = time.time()
    eeg_data = np.array(request.eeg_data, dtype=np.float32)

    # Mock prediction for demo
    n_trials = len(eeg_data)
    n_classes = model.num_classes or 4
    predictions = [0] * n_trials
    probabilities = [[0.25] * n_classes for _ in range(n_trials)]
    for i in range(n_trials):
        predictions[i] = i % n_classes
        probabilities[i][predictions[i]] = 0.7

    latency_ms = (time.time() - start_time) * 1000

    # Save to DB
    async with AsyncSessionLocal() as db:
        job = InferenceJob(
            name=f"Inference_{model.name}",
            model_id=model.id,
            project_id=model.project_id,
            owner_id=current_user["sub"],
            predictions=predictions,
            probabilities=probabilities,
            confidence=max(probabilities[0]) if probabilities else 0,
            latency_ms=latency_ms,
        )
        db.add(job)
        await db.commit()

    return InferenceResponse(
        predictions=predictions,
        probabilities=probabilities,
        class_labels=model.classes or ["Left Fist", "Right Fist", "Both Fists", "Both Feet"],
        latency_ms=round(latency_ms, 2),
        model_used=request.model_name,
    )


@router.get("/jobs", response_model=List[InferenceJobResponse])
async def list_inference_jobs(current_user: dict = Depends(get_current_user)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(InferenceJob).where(InferenceJob.owner_id == current_user["sub"])
        )
        return result.scalars().all()