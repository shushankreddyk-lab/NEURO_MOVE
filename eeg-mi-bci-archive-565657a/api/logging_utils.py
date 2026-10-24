"""
Structured logging utilities for NeuroMove Enterprise.
"""
import logging
import json
import sys
import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any
from pythonjsonlogger import jsonlogger

# Context variables for request/model tracking
request_id_var: ContextVar[str] = ContextVar('request_id', default='')
model_id_var: ContextVar[str] = ContextVar('model_id', default='')
user_id_var: ContextVar[str] = ContextVar('user_id', default='')


class RequestIdFilter(logging.Filter):
    """Add request_id to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get('')
        return True


class ModelContextFilter(logging.Filter):
    """Add model context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        record.model_id = model_id_var.get('')
        record.user_id = user_id_var.get('')
        return True


class StructuredFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with standard fields."""
    
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        super().add_fields(log_record, record, message_dict)
        log_record['timestamp'] = self.formatTime(record, self.datefmt)
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add context fields
        if hasattr(record, 'request_id') and record.request_id:
            log_record['request_id'] = record.request_id
        if hasattr(record, 'model_id') and record.model_id:
            log_record['model_id'] = record.model_id
        if hasattr(record, 'user_id') and record.user_id:
            log_record['user_id'] = record.user_id


def setup_logging(config_path: Optional[str] = None) -> None:
    """Configure structured logging from YAML config."""
    import yaml
    from logging.config import dictConfig
    
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        dictConfig(config)
    else:
        # Default configuration
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S%z'
        ))
        handler.addFilter(RequestIdFilter())
        handler.addFilter(ModelContextFilter())
        
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.handlers = [handler]


def get_logger(name: str) -> logging.Logger:
    """Get logger with structured formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s',
            datefmt='%Y-%m-%dT%H:%M:%S%z'
        ))
        handler.addFilter(RequestIdFilter())
        handler.addFilter(ModelContextFilter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_request_context(request_id: str, model_id: str = '', user_id: str = '') -> None:
    """Set context variables for current request."""
    request_id_var.set(request_id)
    model_id_var.set(model_id)
    user_id_var.set(user_id)


def clear_request_context() -> None:
    """Clear context variables."""
    request_id_var.set('')
    model_id_var.set('')
    user_id_var.set('')


class LoggingMiddleware:
    """ASGI middleware for request logging."""
    
    def __init__(self, app):
        self.app = app
        self.logger = get_logger("neuromove.middleware")
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        request_id = scope.get("headers", {}).get(b"x-request-id", uuid.uuid4().hex[:8]).decode()
        set_request_context(request_id)
        
        start_time = time.time()
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                process_time = time.time() - start_time
                self.logger.info(
                    "HTTP request completed",
                    extra={
                        "request_id": request_id,
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "status_code": message.get("status"),
                        "process_time_ms": round(process_time * 1000, 2)
                    }
                )
            await send(message)
        
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            clear_request_context()


# Context managers for model operations
class ModelOperationContext:
    """Context manager for model operations with automatic logging."""
    
    def __init__(self, operation: str, model_id: str = '', **extra):
        self.operation = operation
        self.model_id = model_id
        self.extra = extra
        self.logger = get_logger("neuromove.model")
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        set_request_context(model_id=self.model_id)
        self.logger.info(f"Starting {self.operation}", extra={
            "operation": self.operation,
            "model_id": self.model_id,
            **self.extra
        })
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = round((time.time() - self.start_time) * 1000, 2)
        if exc_type:
            self.logger.error(f"Failed {self.operation}", extra={
                "operation": self.operation,
                "model_id": self.model_id,
                "duration_ms": duration_ms,
                "error": str(exc_val),
                **self.extra
            })
        else:
            self.logger.info(f"Completed {self.operation}", extra={
                "operation": self.operation,
                "model_id": self.model_id,
                "duration_ms": duration_ms,
                **self.extra
            })
        clear_request_context()
        return False


import time