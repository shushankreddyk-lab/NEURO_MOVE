"""
Dataset management endpoints
"""
from fastapi import APIRouter, Depends
from app.core.security import get_current_user
from app.db.database import AsyncSessionLocal
from app.models.base import Dataset
from app.schemas.base import DatasetResponse, DatasetCreate
from sqlalchemy import select
import uuid

router = APIRouter()


@router.post("", response_model=DatasetResponse)
async def create_dataset(request: DatasetCreate, current_user: dict = Depends(get_current_user)):
    async with AsyncSessionLocal() as db:
        dataset = Dataset(
            name=request.name,
            dataset_type=request.dataset_type,
            subjects=request.subjects,
            channels=request.channels,
            sampling_rate=request.sampling_rate,
            classes=request.classes or [],
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        return dataset


@router.get("", response_model=List[DatasetResponse])
async def list_datasets(current_user: dict = Depends(get_current_user)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Dataset))
        return result.scalars().all()