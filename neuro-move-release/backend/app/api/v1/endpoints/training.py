"""
Training job endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.core.security import get_current_user, require_researcher
from app.db.database import AsyncSessionLocal
from app.models.base import TrainingJob, Model, User
from app.schemas.base import TrainingJobCreate, TrainingJobResponse
from sqlalchemy import select
import uuid, asyncio

router = APIRouter()


@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    request: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(require_researcher),
):
    async with AsyncSessionLocal() as db:
        model_result = await db.execute(select(Model).where(Model.id == request.model_id))
        model = model_result.scalar_one_or_none()
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")

        job = TrainingJob(
            name=request.name or f"Training_{model.name}",
            model_id=request.model_id,
            project_id=model.project_id,
            owner_id=current_user["sub"],
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            kernel_count=request.kernel_count,
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        background_tasks.add_task(run_training, str(job.id))
        return job


async def run_training(job_id: str):
    """Background training task."""
    await asyncio.sleep(5)  # Simulate training


@router.get("/jobs", response_model=List[TrainingJobResponse])
async def list_training_jobs(current_user: dict = Depends(get_current_user)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(TrainingJob).where(TrainingJob.owner_id == current_user["sub"])
        )
        return result.scalars().all()