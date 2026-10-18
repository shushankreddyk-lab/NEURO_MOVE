"""
NeuroMove Enterprise Backend - Core Configuration
"""
from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    # App
    PROJECT_NAME: str = "NeuroMove Enterprise MI-BCI"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Enterprise Motor Imagery BCI Platform"
    API_V1_STR: str = "/api/v1"

    # Security
    SECRET_KEY: str = "neuromove-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://neuromove:neuromove@postgres:5432/neuromove"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # RabbitMQ
    RABBITMQ_URL: str = "amqp://neuromove:neuromove@rabbitmq:5672/"

    # Keycloak
    KEYCLOAK_URL: str = "http://keycloak:8080"
    KEYCLOAK_REALM: str = "neuromove"
    KEYCLOAK_CLIENT_ID: str = "neuromove-backend"
    KEYCLOAK_CLIENT_SECRET: str = "neuromove-backend-secret"

    # Cloudflare Turnstile
    TURNSTILE_SECRET: str = ""

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Models directory
    MODELS_DIR: str = "/app/models"
    NEUROMOVE_ROOT: str = "/app"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()