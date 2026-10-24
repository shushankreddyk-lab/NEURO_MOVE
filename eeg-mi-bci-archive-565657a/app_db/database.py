"""
Database configuration and session management.
"""
import os
from contextlib import contextmanager
from typing import Generator, Optional
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from .models import Base


class DatabaseConfig:
    """Database configuration from environment variables."""

    def __init__(self):
        self.host = os.getenv("APP_DB_HOST", "localhost")
        self.port = int(os.getenv("APP_DB_PORT", "5432"))
        self.database = os.getenv("APP_DB_NAME", "neuromove_app")
        self.user = os.getenv("APP_DB_USER", "neuromove")
        self.password = os.getenv("APP_DB_PASSWORD", "neuromove_secure_pass")
        self.ssl_mode = os.getenv("APP_DB_SSL_MODE", "prefer")

        # Pool settings
        self.pool_size = int(os.getenv("APP_DB_POOL_SIZE", "10"))
        self.max_overflow = int(os.getenv("APP_DB_MAX_OVERFLOW", "20"))
        self.pool_timeout = int(os.getenv("APP_DB_POOL_TIMEOUT", "30"))
        self.pool_recycle = int(os.getenv("APP_DB_POOL_RECYCLE", "3600"))

    @property
    def url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )

    @property
    def async_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )


# Global config instance
_db_config: Optional[DatabaseConfig] = None
_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_db_config() -> DatabaseConfig:
    global _db_config
    if _db_config is None:
        _db_config = DatabaseConfig()
    return _db_config


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        config = get_db_config()
        _engine = create_engine(
            config.url,
            poolclass=QueuePool,
            pool_size=config.pool_size,
            max_overflow=config.max_overflow,
            pool_timeout=config.pool_timeout,
            pool_recycle=config.pool_recycle,
            pool_pre_ping=True,
            echo=os.getenv("APP_DB_ECHO", "false").lower() == "true",
        )
    return _engine


def get_session_factory() -> sessionmaker:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _session_factory


def init_db():
    """Create all tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def drop_db():
    """Drop all tables (use with caution!)."""
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get a database session with automatic commit/rollback."""
    session_factory = get_session_factory()
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database session."""
    with get_db_session() as session:
        yield session


# SQLite for testing
_test_engine: Optional[Engine] = None
_test_session_factory: Optional[sessionmaker] = None


def get_test_db_session() -> Generator[Session, None, None]:
    """Get a test database session (SQLite in-memory)."""
    global _test_engine, _test_session_factory

    if _test_engine is None:
        _test_engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            echo=False,
        )
        Base.metadata.create_all(bind=_test_engine)

    if _test_session_factory is None:
        _test_session_factory = sessionmaker(
            bind=_test_engine,
            autocommit=False,
            autoflush=False,
        )

    session = _test_session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_test_db():
    """Reset test database."""
    global _test_engine, _test_session_factory
    if _test_engine:
        Base.metadata.drop_all(bind=_test_engine)
        Base.metadata.create_all(bind=_test_engine)