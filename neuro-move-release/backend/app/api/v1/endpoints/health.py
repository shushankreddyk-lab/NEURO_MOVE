"""
Health check endpoints
"""
from fastapi import APIRouter
from app.schemas.base import HealthResponse

router = APIRouter()


@router.get("", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=0,
    )