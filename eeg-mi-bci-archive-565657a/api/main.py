"""
Production FastAPI Server for NeuroMove Enterprise MI-BCI.
Provides REST API + WebSocket for real-time EEG inference.
With Cloudflare Turnstile, Rate Limiting, Security Headers, and Enhanced Auth.
"""
import os
import sys
import asyncio
import logging
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Import security module
from security import (
    create_secure_app,
    limiter,
    verify_api_key_enhanced,
    require_admin,
    require_model_management,
    require_inference,
    require_admin_stats,
    verify_ws_token,
    validate_eeg_data_shape,
    TurnstileConfig,
    security_config,
    api_key_manager,
    RateLimitExceeded,
    custom_rate_limit_handler,
    INFERENCE_RATE_LIMIT,
    MODEL_LOAD_RATE_LIMIT,
    ADMIN_RATE_LIMIT,
    HEALTH_RATE_LIMIT,
)

try:
    from minirocket_engine import MiniRocketPipeline
    from config_loader import get_config
    from model_registry import get_registry
    from evaluation import compute_metrics
except ImportError as e:
    print(f"Import warning: {e}")

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("neuromove-api")

# Global state
@dataclass
class ModelState:
    pipeline: Optional[Any] = None
    model_name: str = ""
    model_id: str = ""
    loaded_at: float = 0
    metadata: Dict = None

class AppState:
    def __init__(self):
        self.model: ModelState = ModelState()
        self.active_connections: Dict[str, WebSocket] = {}
        self.inference_count: int = 0
        self.total_latency_ms: float = 0.0
        self.start_time: float = time.time()

app_state = AppState()


# =============================================================================
# Pydantic Models
# =============================================================================

class InferenceRequest(BaseModel):
    data: List[List[List[float]]] = Field(..., description="EEG data [batch, channels, samples]")
    model_id: Optional[str] = None
    return_probabilities: bool = True

    @validator('data')
    def validate_data(cls, v):
        if not v or not v[0] or not v[0][0]:
            raise ValueError("Data must be 3D: [batch, channels, samples]")
        return v


class InferenceResponse(BaseModel):
    prediction: List[str]
    probabilities: Optional[List[List[float]]] = None
    confidence: List[float]
    latency_ms: float
    model_id: str
    request_id: str


class ModelLoadRequest(BaseModel):
    model_name: str = Field(..., description="Model name (e.g., minirocket_latest)")
    model_path: Optional[str] = None


class ModelInfo(BaseModel):
    model_id: str
    model_name: str
    architecture: str
    dataset: str
    subjects: str
    accuracy: float
    file_size_mb: float
    registered_at: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model_loaded: bool
    model_id: Optional[str]
    active_connections: int
    inference_count: int
    avg_latency_ms: float
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    inference_count: int
    total_latency_ms: float
    avg_latency_ms: float
    uptime_seconds: float
    active_connections: int
    model_loaded: bool


# =============================================================================
# Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting NeuroMove Enterprise API")
    # Auto-load default model if available
    try:
        reg = get_registry()
        best = reg.best(architecture="minirocket")
        if best:
            from minirocket_engine import MiniRocketPipeline
            pipeline = MiniRocketPipeline()
            pipeline.load(best["file_path"])
            app_state.model = ModelState(
                pipeline=pipeline,
                model_name=best["model_name"],
                model_id=best["id"],
                loaded_at=time.time(),
                metadata=best
            )
            logger.info(f"Auto-loaded model: {best['id']}")
    except Exception as e:
        logger.warning(f"Could not auto-load model: {e}")
    yield
    logger.info("Shutting down NeuroMove Enterprise API")


# =============================================================================
# FastAPI App with Security Middleware
# =============================================================================

# Create base app
base_app = FastAPI(
    title="NeuroMove Enterprise MI-BCI API",
    description="Production-grade Motor Imagery BCI Inference API with WebSocket support",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Apply all security middleware
app = create_secure_app(base_app)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)


# =============================================================================
# Health & Readiness (with rate limiting)
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
@limiter.limit(HEALTH_RATE_LIMIT)
async def health_check(request: Request):
    uptime = time.time() - app_state.start_time
    avg_latency = app_state.total_latency_ms / app_state.inference_count if app_state.inference_count > 0 else 0
    return HealthResponse(
        status="healthy" if app_state.model.pipeline else "degraded",
        uptime_seconds=uptime,
        model_loaded=app_state.model.pipeline is not None,
        model_id=app_state.model.model_id if app_state.model.pipeline else None,
        active_connections=len(app_state.active_connections),
        inference_count=app_state.inference_count,
        avg_latency_ms=avg_latency
    )


@app.get("/ready", tags=["Health"])
@limiter.limit(HEALTH_RATE_LIMIT)
async def readiness_check(request: Request):
    if app_state.model.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


# =============================================================================
# Metrics
# =============================================================================

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
@limiter.limit(HEALTH_RATE_LIMIT)
async def get_metrics(request: Request, key_info: Dict = Depends(verify_api_key_enhanced)):
    avg_latency = app_state.total_latency_ms / app_state.inference_count if app_state.inference_count > 0 else 0
    return MetricsResponse(
        inference_count=app_state.inference_count,
        total_latency_ms=app_state.total_latency_ms,
        avg_latency_ms=avg_latency,
        uptime_seconds=time.time() - app_state.start_time,
        active_connections=len(app_state.active_connections),
        model_loaded=app_state.model.pipeline is not None
    )


# Prometheus metrics endpoint (no auth for Prometheus scrape)
@app.get("/metrics/prometheus", tags=["Monitoring"])
@limiter.limit(HEALTH_RATE_LIMIT)
async def prometheus_metrics(request: Request):
    avg_latency = app_state.total_latency_ms / app_state.inference_count if app_state.inference_count > 0 else 0
    metrics = f"""# HELP neuromove_inference_total Total inference requests
# TYPE neuromove_inference_total counter
neuromove_inference_total {app_state.inference_count}
# HELP neuromove_latency_ms Average inference latency in milliseconds
# TYPE neuromove_latency_ms gauge
neuromove_latency_ms {avg_latency}
# HELP neuromove_uptime_seconds Server uptime in seconds
# TYPE neuromove_uptime_seconds gauge
neuromove_uptime_seconds {time.time() - app_state.start_time}
# HELP neuromove_model_loaded Model loaded status
# TYPE neuromove_model_loaded gauge
neuromove_model_loaded {1 if app_state.model.pipeline else 0}
# HELP neuromove_active_connections Active WebSocket connections
# TYPE neuromove_active_connections gauge
neuromove_active_connections {len(app_state.active_connections)}
"""
    return JSONResponse(content=metrics, media_type="text/plain")


# =============================================================================
# Model Management
# =============================================================================

@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
@limiter.limit(ADMIN_RATE_LIMIT)
async def list_models(request: Request, key_info: Dict = Depends(require_admin)):
    reg = get_registry()
    models = reg.list()
    return [ModelInfo(**m) for m in models]


@app.post("/models/load", response_model=Dict, tags=["Models"])
@limiter.limit(MODEL_LOAD_RATE_LIMIT)
async def load_model(request: Request, body: ModelLoadRequest, key_info: Dict = Depends(require_model_management)):
    try:
        reg = get_registry()
        model_path = body.model_path
        
        if not model_path:
            model_info = reg.get(body.model_name)
            if not model_info:
                model_info = reg.best(architecture=body.model_name.split("_")[0])
            if not model_info:
                raise HTTPException(404, f"Model {body.model_name} not found")
            model_path = model_info["file_path"]
        
        from minirocket_engine import MiniRocketPipeline
        pipeline = MiniRocketPipeline()
        pipeline.load(model_path)
        
        model_id = model_info["id"] if 'model_info' in locals() else f"{body.model_name}_{uuid.uuid4().hex[:8]}"
        
        app_state.model = ModelState(
            pipeline=pipeline,
            model_name=body.model_name,
            model_id=model_id,
            loaded_at=time.time(),
            metadata=model_info if 'model_info' in locals() else {}
        )
        
        logger.info(f"Loaded model: {model_id}")
        return {"status": "loaded", "model_id": model_id, "model_name": body.model_name}
        
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise HTTPException(500, f"Failed to load model: {str(e)}")


@app.post("/models/unload", tags=["Models"])
@limiter.limit(MODEL_LOAD_RATE_LIMIT)
async def unload_model(request: Request, key_info: Dict = Depends(require_model_management)):
    old_id = app_state.model.model_id
    app_state.model = ModelState()
    logger.info(f"Unloaded model: {old_id}")
    return {"status": "unloaded", "previous_model_id": old_id}


# =============================================================================
# Inference (REST)
# =============================================================================

@app.post("/inference", response_model=InferenceResponse, tags=["Inference"])
@limiter.limit(INFERENCE_RATE_LIMIT)
async def inference(request: Request, body: InferenceRequest, key_info: Dict = Depends(require_inference)):
    if app_state.model.pipeline is None:
        raise HTTPException(503, "No model loaded. Call /models/load first.")
    
    # Validate EEG data shape
    is_valid, error, shape = validate_eeg_data_shape(body.data)
    if not is_valid:
        raise HTTPException(400, f"Invalid EEG data: {error}")
    
    request_id = str(uuid.uuid4())
    start = time.time()
    
    try:
        # Convert to numpy
        X = np.array(body.data, dtype=np.float32)
        
        # Preprocess if needed (ensure 3D)
        if X.ndim == 2:
            X = X[np.newaxis, ...]
        
        # Run inference
        pipeline = app_state.model.pipeline
        probs = pipeline.predict_proba(X)
        preds_idx = np.argmax(probs, axis=1)
        pred_classes = pipeline.classes_[preds_idx] if hasattr(pipeline, 'classes_') else [str(i) for i in preds_idx]
        confidence = np.max(probs, axis=1)
        
        latency_ms = (time.time() - start) * 1000
        
        # Update metrics
        app_state.inference_count += 1
        app_state.total_latency_ms += latency_ms
        
        response = InferenceResponse(
            prediction=pred_classes.tolist(),
            probabilities=probs.tolist() if body.return_probabilities else None,
            confidence=confidence.tolist(),
            latency_ms=latency_ms,
            model_id=app_state.model.model_id,
            request_id=request_id
        )
        
        logger.info(f"Inference: {request_id} | latency={latency_ms:.1f}ms | model={app_state.model.model_id}")
        return response
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(500, f"Inference failed: {str(e)}")


# =============================================================================
# Inference (WebSocket) - for real-time streaming
# =============================================================================

@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    # Verify WebSocket connection
    key_info = await verify_ws_token(websocket)
    if not key_info:
        return
    
    if not api_key_manager.has_scope(key_info, "inference"):
        await websocket.close(code=4003, reason="Insufficient permissions")
        return
    
    await websocket.accept()
    conn_id = str(uuid.uuid4())
    app_state.active_connections[conn_id] = websocket
    logger.info(f"WebSocket connected: {conn_id} (total: {len(app_state.active_connections)})")
    
    try:
        while True:
            # Receive data
            data = await websocket.receive_json()
            
            if app_state.model.pipeline is None:
                await websocket.send_json({"error": "No model loaded"})
                continue
            
            try:
                X = np.array(data.get("data", []), dtype=np.float32)
                if X.ndim == 2:
                    X = X[np.newaxis, ...]
                
                # Validate
                is_valid, error, _ = validate_eeg_data_shape(X.tolist())
                if not is_valid:
                    await websocket.send_json({"error": f"Invalid data: {error}"})
                    continue
                
                start = time.time()
                pipeline = app_state.model.pipeline
                probs = pipeline.predict_proba(X)
                preds_idx = np.argmax(probs, axis=1)
                pred_classes = pipeline.classes_[preds_idx] if hasattr(pipeline, 'classes_') else [str(i) for i in preds_idx]
                confidence = np.max(probs, axis=1)
                latency_ms = (time.time() - start) * 1000
                
                # Update metrics
                app_state.inference_count += 1
                app_state.total_latency_ms += latency_ms
                
                await websocket.send_json({
                    "request_id": data.get("request_id", str(uuid.uuid4())),
                    "prediction": pred_classes.tolist(),
                    "probabilities": probs.tolist(),
                    "confidence": confidence.tolist(),
                    "latency_ms": latency_ms,
                    "model_id": app_state.model.model_id
                })
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {conn_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        app_state.active_connections.pop(conn_id, None)


# =============================================================================
# Admin endpoints
# =============================================================================

@app.get("/admin/stats", tags=["Admin"])
@limiter.limit(ADMIN_RATE_LIMIT)
async def admin_stats(request: Request, key_info: Dict = Depends(require_admin_stats)):
    reg = get_registry()
    models = reg.list()
    return {
        "server": {
            "uptime_seconds": time.time() - app_state.start_time,
            "inference_count": app_state.inference_count,
            "avg_latency_ms": app_state.total_latency_ms / app_state.inference_count if app_state.inference_count > 0 else 0,
            "active_connections": len(app_state.active_connections),
            "model_loaded": app_state.model.pipeline is not None,
            "current_model": app_state.model.model_id
        },
        "registry": {
            "total_models": len(models),
            "architectures": list(set(m["architecture"] for m in models)),
            "datasets": list(set(m["dataset"] for m in models))
        }
    }


# =============================================================================
# Turnstile Configuration Endpoint
# =============================================================================

@app.get("/config/turnstile", tags=["Config"])
async def get_turnstile_config():
    """Get Turnstile site key for frontend integration."""
    return {
        "site_key": security_config.TURNSTILE_SITE_KEY,
        "enabled": security_config.TURNSTILE_ENABLED
    }


# =============================================================================
# Error handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_config=None
    )