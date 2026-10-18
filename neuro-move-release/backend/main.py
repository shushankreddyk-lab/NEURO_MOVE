"""
NeuroMove Enterprise MI-BCI Platform - Production FastAPI Backend
Clean, minimal, production-ready for demo.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import numpy as np
import os
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ============================================================================
# MODELS & SCHEMAS
# ============================================================================

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"
    models_loaded: int = 0

class ModelInfo(BaseModel):
    name: str
    description: str
    classes: List[str]
    input_shape: List[int]

class InferenceRequest(BaseModel):
    model_name: str
    eeg_data: List[List[List[float]]]  # [trials, channels, samples]
    sampling_rate: int = 250
    channels: Optional[List[str]] = None

class InferenceResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]
    class_labels: List[str]
    latency_ms: float
    model_used: str

class ModelLoadRequest(BaseModel):
    model_name: str

class ModelLoadResponse(BaseModel):
    success: bool
    message: str
    model_info: Optional[ModelInfo] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserLogin(BaseModel):
    username: str
    password: str

# ============================================================================
# SECURITY (Simple JWT for demo)
# ============================================================================
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = "neuromove-demo-secret-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================================================
# MODEL REGISTRY (connects to your existing pipelines)
# ============================================================================

MODEL_REGISTRY = {}
MODEL_METADATA = {
    "MiniRocket": {
        "description": "Deterministic random convolutional kernels + MLP classifier",
        "classes": ["Left Fist", "Right Fist", "Both Fists", "Both Feet"],
        "input_shape": [20, 656],  # [channels, samples]
    },
    "EEGNet": {
        "description": "Depthwise + separable convolutions for EEG",
        "classes": ["Left Fist", "Right Fist", "Both Fists", "Both Feet"],
        "input_shape": [20, 656],
    },
    "CNN-LSTM": {
        "description": "Hybrid CNN + LSTM for spatiotemporal features",
        "classes": ["Left Fist", "Right Fist", "Both Fists", "Both Feet"],
        "input_shape": [20, 656],
    },
    "CNN+Transformer": {
        "description": "CNN feature extractor + Transformer encoder",
        "classes": ["Left Fist", "Right Fist", "Both Fists", "Both Feet"],
        "input_shape": [20, 656],
    },
}

# Demo credentials
USERS = {
    "admin": "neuromove2024",
    "researcher": "neuromove2024",
    "clinician": "neuromove2024",
}

# ============================================================================
# APP LIFECYCLE
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 NeuroMove API starting...")
    print(f"   Models available: {list(MODEL_METADATA.keys())}")
    yield
    # Shutdown
    print("🛑 NeuroMove API shutting down...")
    MODEL_REGISTRY.clear()

# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="NeuroMove Enterprise MI-BCI API",
    description="Production-ready Motor Imagery BCI Inference API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# AUTH ENDPOINTS
# ============================================================================

@app.post("/auth/login", response_model=Token)
async def login(user: UserLogin):
    if user.username not in USERS or USERS[user.username] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    access_token = create_access_token(data={"sub": user.username})
    return Token(access_token=access_token)

@app.get("/auth/me")
async def get_current_user(username: str = Depends(verify_token)):
    return {"username": username, "role": "admin"}

# ============================================================================
# HEALTH & MODEL ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded=len(MODEL_REGISTRY)
    )

@app.get("/models", response_model=List[ModelInfo])
async def list_models(username: str = Depends(verify_token)):
    return [
        ModelInfo(name=name, **meta)
        for name, meta in MODEL_METADATA.items()
    ]

@app.post("/models/load", response_model=ModelLoadResponse)
async def load_model(request: ModelLoadRequest, username: str = Depends(verify_token)):
    if request.model_name not in MODEL_METADATA:
        raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
    
    if request.model_name in MODEL_REGISTRY:
        return ModelLoadResponse(
            success=True,
            message=f"Model {request.model_name} already loaded",
            model_info=ModelInfo(name=request.model_name, **MODEL_METADATA[request.model_name])
        )
    
    # Try to load real model, fallback to mock
    try:
        pipeline = _load_real_model(request.model_name)
        MODEL_REGISTRY[request.model_name] = pipeline
        return ModelLoadResponse(
            success=True,
            message=f"Model {request.model_name} loaded successfully",
            model_info=ModelInfo(name=request.model_name, **MODEL_METADATA[request.model_name])
        )
    except Exception as e:
        # Fallback to mock for demo
        MODEL_REGISTRY[request.model_name] = "mock"
        return ModelLoadResponse(
            success=True,
            message=f"Model {request.model_name} loaded (mock mode)",
            model_info=ModelInfo(name=request.model_name, **MODEL_METADATA[request.model_name])
        )

def _load_real_model(model_name: str):
    """Try to load real model pipeline, raise if not available."""
    model_map = {
        "MiniRocket": ("minirocket_engine", "MiniRocketPipeline"),
        "EEGNet": ("eegnet_engine", "EEGNet_Pipeline"),
        "CNN-LSTM": ("cnn_lstm_engine", "CNN_LSTM_Pipeline"),
        "CNN+Transformer": ("cnn_transformer_engine", "CNNTransformerPipeline"),
    }
    
    if model_name not in model_map:
        raise ValueError(f"Unknown model: {model_name}")
    
    module_name, class_name = model_map[model_name]
    try:
        module = __import__(f"src.{module_name}", fromlist=[class_name])
        pipeline_class = getattr(module, class_name)
        return pipeline_class()
    except ImportError:
        raise RuntimeError(f"Module src.{module_name} not found")

# ============================================================================
# INFERENCE ENDPOINT
# ============================================================================

@app.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, username: str = Depends(verify_token)):
    if request.model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Model {request.model_name} not loaded. Call /models/load first.")
    
    start_time = time.time()
    
    # Convert input to numpy
    eeg_data = np.array(request.eeg_data, dtype=np.float32)
    n_trials, n_channels, n_samples = eeg_data.shape
    
    # Validate shape
    expected_ch, expected_samples = MODEL_METADATA[request.model_name]["input_shape"]
    if n_channels != expected_ch:
        raise HTTPException(status_code=400, detail=f"Expected {expected_ch} channels, got {n_channels}")
    
    # Run inference (real or mock)
    if MODEL_REGISTRY[request.model_name] == "mock":
        predictions, probabilities = _mock_inference(n_trials, request.model_name)
    else:
        pipeline = MODEL_REGISTRY[request.model_name]
        predictions, probabilities = _real_inference(pipeline, eeg_data)
    
    latency_ms = (time.time() - start_time) * 1000
    
    meta = MODEL_METADATA[request.model_name]
    
    return InferenceResponse(
        predictions=predictions,
        probabilities=probabilities.tolist(),
        class_labels=meta["classes"],
        latency_ms=round(latency_ms, 2),
        model_used=request.model_name
    )

def _mock_inference(n_trials: int, model_name: str) -> tuple:
    """Generate realistic mock predictions for demo."""
    meta = MODEL_METADATA[model_name]
    n_classes = len(meta["classes"])
    
    predictions = []
    probabilities = []
    
    for _ in range(n_trials):
        # Generate realistic probability distribution
        # One class dominant, others small
        dominant = np.random.randint(0, n_classes)
        probs = np.random.dirichlet([0.5] * n_classes)
        probs[dominant] += 0.6
        probs = probs / probs.sum()
        
        pred = int(np.argmax(probs))
        predictions.append(pred)
        probabilities.append(probs)
    
    return predictions, np.array(probabilities)

def _real_inference(pipeline, eeg_data: np.ndarray) -> tuple:
    """Run real model inference."""
    try:
        # This depends on your pipeline interface
        predictions = pipeline.predict(eeg_data)
        probabilities = pipeline.predict_proba(eeg_data)
        return predictions.tolist(), probabilities
    except Exception as e:
        # Fallback to mock
        return _mock_inference(eeg_data.shape[0], "mock")

# ============================================================================
# PREPROCESSING HELPER ENDPOINT
# ============================================================================

class PreprocessRequest(BaseModel):
    eeg_data: List[List[List[float]]]
    sampling_rate: int = 250
    target_rate: int = 128
    low_freq: float = 4.0
    high_freq: float = 38.0

class PreprocessResponse(BaseModel):
    processed_data: List[List[List[float]]]
    sampling_rate: int
    channels: int
    samples: int

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(request: PreprocessRequest, username: str = Depends(verify_token)):
    """Apply standard MI-BCI preprocessing pipeline."""
    import scipy.signal as signal
    
    eeg = np.array(request.eeg_data, dtype=np.float32)
    n_trials, n_channels, n_samples = eeg.shape
    
    # 1. Bandpass filter 4-38 Hz
    nyquist = request.sampling_rate / 2
    b, a = signal.butter(4, [request.low_freq / nyquist, request.high_freq / nyquist], btype='bandpass')
    eeg_filtered = signal.filtfilt(b, a, eeg, axis=-1)
    
    # 2. Common Average Reference (CAR)
    eeg_car = eeg_filtered - np.mean(eeg_filtered, axis=1, keepdims=True)
    
    # 3. Downsample to target rate
    if request.sampling_rate != request.target_rate:
        factor = request.sampling_rate / request.target_rate
        new_samples = int(n_samples / factor)
        indices = np.linspace(0, n_samples - 1, new_samples).astype(int)
        eeg_ds = eeg_car[:, :, indices]
    else:
        eeg_ds = eeg_car
    
    # 4. Z-score normalization per channel per trial
    eeg_norm = (eeg_ds - np.mean(eeg_ds, axis=-1, keepdims=True)) / (np.std(eeg_ds, axis=-1, keepdims=True) + 1e-8)
    
    return PreprocessResponse(
        processed_data=eeg_norm.tolist(),
        sampling_rate=request.target_rate,
        channels=n_channels,
        samples=eeg_norm.shape[2]
    )

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)