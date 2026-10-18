"""
NeuroMove Enterprise Backend - API v1 Router
"""
from fastapi import APIRouter

api_router = APIRouter()

from app.api.v1.endpoints import health, auth, models, training, inference, datasets, audit

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(training.router, prefix="/training", tags=["training"])
api_router.include_router(inference.router, prefix="/inference", tags=["inference"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(audit.router, prefix="/audit", tags=["audit"])