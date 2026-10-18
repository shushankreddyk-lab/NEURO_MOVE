"""
NeuroMove Enterprise Backend - SQLAlchemy Models
"""
import uuid
from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Column, String, Integer, DateTime, ForeignKey, Text, Boolean, Float, Enum
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from app.db.database import Base
import enum


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    RESEARCHER = "researcher"
    CLINICIAN = "clinician"
    VIEWER = "viewer"


class ProjectStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class TrainingStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class InferenceStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetType(str, enum.Enum):
    PHYSIONET = "physionet"
    BCI_IV_2A = "bci_iv_2a"
    HIGH_GAMMA = "high_gamma"
    KAYA = "kaya"
    WAY_EEG_GAL = "way_eeg_gal"
    NEMAR = "nemar"


class ModelArchitecture(str, enum.Enum):
    MINIROCKET = "minirocket"
    EEGNET = "eegnet"
    CNN_LSTM = "cnn_lstm"
    CNN_TRANSFORMER = "cnn_transformer"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(255), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    keycloak_id = Column(String(255), unique=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    projects = relationship("Project", back_populates="owner")
    training_jobs = relationship("TrainingJob", back_populates="owner")
    inference_jobs = relationship("InferenceJob", back_populates="owner")


class Project(Base):
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(Enum(ProjectStatus), default=ProjectStatus.ACTIVE, nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    dataset_type = Column(Enum(DatasetType), nullable=False)
    channels = Column(Integer, nullable=True)
    sampling_rate = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    owner = relationship("User", back_populates="projects")
    training_jobs = relationship("TrainingJob", back_populates="project")
    inference_jobs = relationship("InferenceJob", back_populates="project")
    models = relationship("Model", back_populates="project")


class Model(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    architecture = Column(Enum(ModelArchitecture), nullable=False)
    version = Column(String(50), default="1.0", nullable=False)
    file_path = Column(String(500), nullable=True)
    num_classes = Column(Integer, nullable=True)
    input_shape = Column(JSONB, nullable=True)
    classes = Column(JSONB, nullable=True)
    accuracy = Column(Float, nullable=True)
    training_time = Column(Float, nullable=True)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    owner = relationship("User")
    project = relationship("Project", back_populates="models")
    training_jobs = relationship("TrainingJob", back_populates="model")


class TrainingJob(Base):
    __tablename__ = "training_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=True)
    status = Column(Enum(TrainingStatus), default=TrainingStatus.PENDING, nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    epochs = Column(Integer, default=10, nullable=False)
    batch_size = Column(Integer, default=32, nullable=False)
    learning_rate = Column(Float, default=0.001, nullable=False)
    kernel_count = Column(Integer, default=10000, nullable=True)
    result_path = Column(String(500), nullable=True)
    accuracy = Column(Float, nullable=True)
    training_time = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    error_message = Column(Text, nullable=True)

    # Relationships
    model = relationship("Model", back_populates="training_jobs")
    project = relationship("Project", back_populates="training_jobs")
    owner = relationship("User", back_populates="training_jobs")


class InferenceJob(Base):
    __tablename__ = "inference_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=True)
    status = Column(Enum(InferenceStatus), default=InferenceStatus.PENDING, nullable=False)
    model_id = Column(UUID(as_uuid=True), ForeignKey("models.id"), nullable=False)
    project_id = Column(UUID(as_uuid=True), ForeignKey("projects.id"), nullable=False)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    input_shape = Column(JSONB, nullable=True)
    predictions = Column(JSONB, nullable=True)
    probabilities = Column(JSONB, nullable=True)
    confidence = Column(Float, nullable=True)
    latency_ms = Column(Float, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    error_message = Column(Text, nullable=True)

    # Relationships
    model = relationship("Model")
    project = relationship("Project", back_populates="inference_jobs")
    owner = relationship("User", back_populates="inference_jobs")


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    action = Column(String(255), nullable=False)
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(255), nullable=True)
    details = Column(JSONB, nullable=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    user = relationship("User")