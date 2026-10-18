"""
Model management endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.core.security import get_current_user, require_researcher
from app.schemas.base import ModelInfo, ModelResponse, ModelCreate
from app.db.database import AsyncSessionLocal
from app.models.base import Model, User, Project
from sqlalchemy import select
from app.core.config import settings
import os, json

router = APIRouter()


@router.get("", response_model=List[ModelInfo])
async def list_models(current_user: dict = Depends(get_current_user)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Model).where(Model.owner_id == current_user["sub"]))
        models = result.scalars().all()
        return [
            ModelInfo(
                name=m.name,
                description=m.architecture,
                classes=m.classes or [],
                input_shape=m.input_shape or {},
            )
            for m in models
        ]


@router.post("/load", response_model=ModelResponse)
async def load_model(request: ModelCreate, current_user: dict = Depends(get_current_user)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Model).where(Model.name == request.name))
        model = result.scalar_one_or_none()
        if not model:
            model = Model(
                name=request.name,
                architecture=request.architecture,
                num_classes=request.num_classes or 4,
                input_shape=request.input_shape or {},
                classes=request.classes or ["Left Fist", "Right Fist", "Both Fists", "Both Feet"],
                owner_id=current_user["sub"],
                project_id=current_user.get("project_id", str(model.id)),
            )
            db.add(model)
            await db.commit()
            await db.refresh(model)
        return model