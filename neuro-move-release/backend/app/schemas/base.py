"""
NeuroMove Enterprise Backend - Pydantic Schemas
"""
from pydantic import BaseModel, EmailStr, ConfigDict
from typing import List, Optional, Any
from datetime import datetime
from uuid import UUID


# ==================== USER SCHEMAS ====================

class UserBase(BaseModel):
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    role: str = "viewer"
    is_active: bool = True


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    keycloak_id: Optional[str] = None
    email_verified: bool = False
    is_superuser: bool = False
    created_at: datetime
    updated_at: datetime


# ==================== AUTH SCHEMAS ====================

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class UserLoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ==================== PROJECT SCHEMAS ====================

class ProjectBase(BaseModel):
    name: str
    description: Optional[str] = None
    dataset_type: str
    channels: Optional[int] = None
    sampling_rate: Optional[int] = None


class ProjectCreate(ProjectBase):
    pass


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None


class ProjectResponse(ProjectBase):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    status: str
    owner_id: UUID
    created_at: datetime
    updated_at: datetime


# ==================== MODEL SCHEMAS ====================

class ModelBase(BaseModel):
    name: str
    architecture: str
    num_classes: Optional[int] = None
    input_shape: Optional[dict] = None
    classes: Optional[list] = None


class ModelCreate(ModelBase):
    pass


class ModelResponse(ModelBase):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    version: str
    file_path: Optional[str] = None
    accuracy: Optional[float] = None
    training_time: Optional[float] = None
    owner_id: UUID
    project_id: UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime


# ==================== TRAINING JOB SCHEMAS ====================

class TrainingJobBase(BaseModel):
    name: Optional[str] = None
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    kernel_count: Optional[int] = None


class TrainingJobCreate(TrainingJobBase):
    model_id: UUID


class TrainingJobResponse(TrainingJobBase):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    status: str
    model_id: UUID
    project_id: UUID
    owner_id: UUID
    result_path: Optional[str] = None
    accuracy: Optional[float] = None
    training_time: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# ==================== INFERENCE SCHEMAS ====================

class InferenceBase(BaseModel):
    model_name: str
    eeg_data: List[List[List[float]]]  # [trials, channels, timepoints]
    sampling_rate: int = 250
    channels: Optional[List[str]] = None


class InferenceRequest(InferenceBase):
    pass


class InferenceResponse(BaseModel):
    predictions: List[int]
    probabilities: List[List[float]]
    class_labels: List[str]
    latency_ms: float
    model_used: str


class InferenceJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: Optional[str] = None
    status: str
    model_id: UUID
    project_id: UUID
    owner_id: UUID
    predictions: Optional[list] = None
    probabilities: Optional[list] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


# ==================== DATASET SCHEMAS ====================

class DatasetBase(BaseModel):
    name: str
    dataset_type: str
    subjects: Optional[int] = None
    channels: Optional[int] = None
    sampling_rate: Optional[int] = None
    classes: Optional[list] = None


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    created_at: datetime


# ==================== AUDIT LOG SCHEMAS ====================

class AuditLogResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    user_id: Optional[UUID] = None
    action: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    details: Optional[dict] = None
    ip_address: Optional[str] = None
    created_at: datetime