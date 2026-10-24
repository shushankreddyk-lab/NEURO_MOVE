"""
Security Module for NeuroMove Enterprise API.
Includes: Cloudflare Turnstile, Rate Limiting, Security Headers, Input Validation.
"""
import os
import time
import hmac
import hashlib
import httpx
import logging
from typing import Optional, Dict, Callable
from functools import wraps

from fastapi import Request, Response, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
import slowapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger("neuromove.security")

# =============================================================================
# Cloudflare Turnstile Configuration
# =============================================================================

class TurnstileConfig:
    """Cloudflare Turnstile configuration."""
    def __init__(
        self,
        site_key: str = None,
        secret_key: str = None,
        verify_url: str = "https://challenges.cloudflare.com/turnstile/v0/siteverify",
        enabled: bool = True
    ):
        self.site_key = site_key or os.getenv("TURNSTILE_SITE_KEY", "")
        self.secret_key = secret_key or os.getenv("TURNSTILE_SECRET_KEY", "")
        self.verify_url = verify_url
        self.enabled = enabled and bool(self.secret_key)
    
    async def verify(self, token: str, remote_ip: str = None) -> bool:
        """Verify Turnstile token with Cloudflare."""
        if not self.enabled:
            return True  # Skip if disabled
        
        if not token:
            return False
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                data = {
                    "secret": self.secret_key,
                    "response": token,
                }
                if remote_ip:
                    data["remoteip"] = remote_ip
                
                response = await client.post(self.verify_url, data=data)
                result = response.json()
                
                success = result.get("success", False)
                if not success:
                    logger.warning(f"Turnstile verification failed: {result.get('error-codes', [])}")
                return success
        except Exception as e:
            logger.error(f"Turnstile verification error: {e}")
            # Fail open in case of Cloudflare issues (configurable)
            return os.getenv("TURNSTILE_FAIL_OPEN", "false").lower() == "true"


turnstile = TurnstileConfig()


# =============================================================================
# Rate Limiting with slowapi
# =============================================================================

# Initialize limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100/minute", "1000/hour"],
    storage_uri=os.getenv("RATE_LIMIT_STORAGE", "memory://"),
    strategy="fixed-window"
)

# Custom rate limit exceeded handler
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": f"Too many requests. Limit: {exc.detail}",
            "retry_after": 60
        },
        headers={"Retry-After": "60"}
    )


def get_rate_limiter() -> Limiter:
    """Get the global rate limiter instance."""
    return limiter


# Rate limit decorators for specific endpoints
def rate_limit(limit: str, key_func: Callable = None):
    """Decorator for custom rate limits on specific endpoints."""
    def decorator(func):
        @wraps(func)
        @limiter.limit(limit, key_func=key_func)
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Pre-configured rate limits
INFERENCE_RATE_LIMIT = "30/minute"
MODEL_LOAD_RATE_LIMIT = "5/minute"
ADMIN_RATE_LIMIT = "20/minute"
HEALTH_RATE_LIMIT = "120/minute"
WS_RATE_LIMIT = "50/minute"


# =============================================================================
# Security Headers Middleware
# =============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    def __init__(
        self,
        app: ASGIApp,
        csp_policy: str = None,
        hsts_max_age: int = 31536000,  # 1 year
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: str = None
    ):
        super().__init__(app)
        self.csp_policy = csp_policy or (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; "
            "img-src 'self' data: https:; "
            "connect-src 'self' wss: https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        self.hsts_max_age = hsts_max_age
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy or (
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), "
            "magnetometer=(), microphone=(), payment=(), usb=()"
        )
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = self.referrer_policy
        response.headers["Permissions-Policy"] = self.permissions_policy
        response.headers["Content-Security-Policy"] = self.csp_policy
        
        # HSTS (only for HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = f"max-age={self.hsts_max_age}; includeSubDomains; preload"
        
        # Remove server header
        response.headers.pop("Server", None)
        
        # Custom headers
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Request-ID"] = request.headers.get("X-Request-ID", "")
        
        return response


# =============================================================================
# Turnstile Middleware
# =============================================================================

class TurnstileMiddleware(BaseHTTPMiddleware):
    """Validate Cloudflare Turnstile token on protected endpoints."""
    
    # Endpoints that require Turnstile verification
    PROTECTED_PATHS = {
        "/inference",
        "/ws/inference",
        "/models/load",
        "/models/unload",
    }
    
    # Methods that require verification
    PROTECTED_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
    
    def __init__(self, app: ASGIApp, turnstile: TurnstileConfig = None):
        super().__init__(app)
        self.turnstile = turnstile or TurnstileConfig()
    
    def _requires_turnstile(self, request: Request) -> bool:
        """Check if endpoint requires Turnstile verification."""
        path = request.url.path
        method = request.method
        
        # Check exact path match
        if path in self.PROTECTED_PATHS and method in self.PROTECTED_METHODS:
            return True
        
        # Check prefix match for websocket
        if path.startswith("/ws/") and method == "GET":
            return True
        
        return False
    
    def _extract_turnstile_token(self, request: Request) -> Optional[str]:
        """Extract Turnstile token from request."""
        # Check header
        token = request.headers.get("X-Turnstile-Token")
        if token:
            return token
        
        # Check query param
        token = request.query_params.get("cf-turnstile-response")
        if token:
            return token
        
        # Check form data (for multipart)
        return None
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not self._requires_turnstile(request):
            return await call_next(request)
        
        # Extract token
        token = self._extract_turnstile_token(request)
        client_ip = request.client.host if request.client else None
        
        # Verify token
        if not await self.turnstile.verify(token, client_ip):
            logger.warning(f"Turnstile verification failed for {client_ip} on {request.url.path}")
            return JSONResponse(
                status_code=403,
                content={
                    "error": "Human verification required",
                    "detail": "Please complete the Turnstile challenge",
                    "turnstile_site_key": self.turnstile.site_key
                }
            )
        
        return await call_next(request)


# =============================================================================
# Request Validation Middleware
# =============================================================================

class RequestValidationMiddleware(BaseHTTPMiddleware):
    """Validate request size, content type, and sanitize input."""
    
    MAX_REQUEST_SIZE = 50 * 1024 * 1024  # 50 MB
    ALLOWED_CONTENT_TYPES = {
        "application/json",
        "multipart/form-data",
        "application/octet-stream"
    }
    
    def __init__(self, app: ASGIApp, max_size: int = None):
        super().__init__(app)
        self.max_size = max_size or self.MAX_REQUEST_SIZE
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check content length
        content_length = request.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > self.max_size:
                    return JSONResponse(
                        status_code=413,
                        content={"error": "Request too large", "max_size_mb": self.max_size // (1024*1024)}
                    )
            except ValueError:
                pass
        
        # Check content type for POST/PUT/PATCH
        if request.method in {"POST", "PUT", "PATCH"}:
            content_type = request.headers.get("Content-Type", "").split(";")[0].strip()
            if content_type and content_type not in self.ALLOWED_CONTENT_TYPES:
                return JSONResponse(
                    status_code=415,
                    content={"error": "Unsupported media type", "allowed": list(self.ALLOWED_CONTENT_TYPES)}
                )
        
        return await call_next(request)


# =============================================================================
# Input Sanitization Utilities
# =============================================================================

def sanitize_string(value: str, max_length: int = 1000) -> str:
    """Sanitize string input."""
    if not isinstance(value, str):
        return str(value)[:max_length]
    
    # Remove null bytes and control characters
    sanitized = value.replace('\x00', '').replace('\r', '').replace('\n', ' ')
    return sanitized[:max_length]


def validate_eeg_data_shape(data: list, expected_dims: tuple = (None, None, None)) -> tuple:
    """
    Validate EEG data shape.
    Returns (is_valid, error_message, shape)
    """
    try:
        import numpy as np
        arr = np.array(data, dtype=np.float32)
        
        if arr.ndim != 3:
            return False, f"Expected 3D array [batch, channels, samples], got {arr.ndim}D", arr.shape
        
        batch, channels, samples = arr.shape
        
        # Validate dimensions
        if expected_dims[0] and batch != expected_dims[0]:
            return False, f"Batch size mismatch: expected {expected_dims[0]}, got {batch}", arr.shape
        if expected_dims[1] and channels != expected_dims[1]:
            return False, f"Channel count mismatch: expected {expected_dims[1]}, got {channels}", arr.shape
        if expected_dims[2] and samples != expected_dims[2]:
            return False, f"Sample count mismatch: expected {expected_dims[2]}, got {samples}", arr.shape
        
        # Check for NaN/Inf
        if np.isnan(arr).any() or np.isinf(arr).any():
            return False, "Data contains NaN or Inf values", arr.shape
        
        # Reasonable bounds check (EEG typically in microvolts)
        if arr.max() > 1e6 or arr.min() < -1e6:
            return False, "Values outside reasonable EEG range", arr.shape
        
        return True, None, arr.shape
    except Exception as e:
        return False, f"Validation error: {str(e)}", None


# =============================================================================
# API Key Management
# =============================================================================

class APIKeyManager:
    """Manage API keys with rotation and scoping."""
    
    def __init__(self):
        self.keys: Dict[str, Dict] = {}
        self._load_from_env()
    
    def _load_from_env(self):
        """Load keys from environment variables."""
        # Format: API_KEY_<NAME>=<KEY>:<ROLE>:<SCOPES>
        # Example: API_KEY_ADMIN=sk_admin_abc123:admin:inference,admin,models
        for key, value in os.environ.items():
            if key.startswith("API_KEY_"):
                parts = value.split(":")
                if len(parts) >= 2:
                    api_key = parts[0]
                    role = parts[1]
                    scopes = parts[2].split(",") if len(parts) > 2 else ["inference"]
                    name = key[8:].lower()  # Remove "API_KEY_"
                    self.keys[api_key] = {
                        "name": name,
                        "role": role,
                        "scopes": scopes,
                        "created_at": time.time()
                    }
        
        # Fallback to defaults if none configured
        if not self.keys:
            self.keys = {
                os.getenv("API_KEY_ADMIN", "neuromove-enterprise"): {
                    "name": "admin", "role": "admin", "scopes": ["inference", "admin", "models", "admin_stats"]
                },
                os.getenv("API_KEY_CLIENT", "neuromove-client"): {
                    "name": "client", "role": "client", "scopes": ["inference"]
                }
            }
    
    def verify(self, api_key: str) -> Optional[Dict]:
        """Verify API key and return key info."""
        return self.keys.get(api_key)
    
    def has_scope(self, key_info: Dict, required_scope: str) -> bool:
        """Check if key has required scope."""
        return required_scope in key_info.get("scopes", [])
    
    def rotate_key(self, old_key: str, new_key: str) -> bool:
        """Rotate API key."""
        if old_key in self.keys:
            info = self.keys.pop(old_key)
            self.keys[new_key] = info
            return True
        return False


api_key_manager = APIKeyManager()


# =============================================================================
# Enhanced Authentication Dependency
# =============================================================================

async def verify_api_key_enhanced(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer(auto_error=False))
) -> Dict:
    """Enhanced API key verification with scopes."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    key_info = api_key_manager.verify(credentials.credentials)
    if not key_info:
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    # Add key info to request state for downstream use
    request.state.api_key_info = key_info
    return key_info


def require_scope(required_scope: str):
    """Dependency factory for scope-based authorization."""
    async def scope_checker(key_info: Dict = Depends(verify_api_key_enhanced)):
        if not api_key_manager.has_scope(key_info, required_scope):
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required scope: {required_scope}"
            )
        return key_info
    return scope_checker


# Scope-specific dependencies
require_admin = require_scope("admin")
require_model_management = require_scope("models")
require_inference = require_scope("inference")
require_admin_stats = require_scope("admin_stats")


# =============================================================================
# Security Configuration
# =============================================================================

class SecurityConfig:
    """Centralized security configuration."""
    
    # CORS
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://localhost:3000").split(",")
    CORS_ALLOW_CREDENTIALS = True
    CORS_ALLOW_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS = ["*"]
    CORS_EXPOSE_HEADERS = ["X-Request-ID", "X-RateLimit-Limit", "X-RateLimit-Remaining"]
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "100/minute")
    
    # Turnstile
    TURNSTILE_ENABLED = os.getenv("TURNSTILE_ENABLED", "true").lower() == "true"
    TURNSTILE_SITE_KEY = os.getenv("TURNSTILE_SITE_KEY", "")
    TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY", "")
    TURNSTILE_FAIL_OPEN = os.getenv("TURNSTILE_FAIL_OPEN", "false").lower() == "true"
    
    # Security Headers
    CSP_POLICY = os.getenv("CSP_POLICY", "")
    HSTS_MAX_AGE = int(os.getenv("HSTS_MAX_AGE", "31536000"))
    
    # Request Validation
    MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE_MB", "50")) * 1024 * 1024
    
    # API Keys
    API_KEY_ADMIN = os.getenv("API_KEY_ADMIN", "")
    API_KEY_CLIENT = os.getenv("API_KEY_CLIENT", "")


security_config = SecurityConfig()


# =============================================================================
# Middleware Factory
# =============================================================================

def create_secure_app(base_app: FastAPI) -> FastAPI:
    """Add all security middleware to FastAPI app."""
    
    # 1. Security Headers (outermost)
    base_app.add_middleware(SecurityHeadersMiddleware)
    
    # 2. Request Validation
    base_app.add_middleware(RequestValidationMiddleware)
    
    # 3. Turnstile Verification
    if security_config.TURNSTILE_ENABLED:
        base_app.add_middleware(TurnstileMiddleware, turnstile=turnstile)
    
    # 4. Rate Limiting (via slowapi - add exception handler)
    base_app.state.limiter = limiter
    base_app.add_exception_handler(RateLimitExceeded, custom_rate_limit_handler)
    
    # 5. CORS (configured last, applied first)
    base_app.add_middleware(
        CORSMiddleware,
        allow_origins=security_config.CORS_ORIGINS,
        allow_credentials=security_config.CORS_ALLOW_CREDENTIALS,
        allow_methods=security_config.CORS_ALLOW_METHODS,
        allow_headers=security_config.CORS_ALLOW_HEADERS,
        expose_headers=security_config.CORS_EXPOSE_HEADERS,
    )
    
    # 6. GZip (innermost)
    base_app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    return base_app


# =============================================================================
# WebSocket Security Helpers
# =============================================================================

async def verify_ws_token(websocket: WebSocket) -> Dict:
    """Verify API key and optional Turnstile for WebSocket connections."""
    # Get API key from query param or header
    api_key = websocket.query_params.get("api_key") or websocket.headers.get("Authorization", "").replace("Bearer ", "")
    
    if not api_key:
        await websocket.close(code=4001, reason="Missing API key")
        return None
    
    key_info = api_key_manager.verify(api_key)
    if not key_info:
        await websocket.close(code=4003, reason="Invalid API key")
        return None
    
    # Check Turnstile for WebSocket (optional - can be required for initial connection)
    turnstile_token = websocket.query_params.get("cf-turnstile-response")
    if security_config.TURNSTILE_ENABLED and not await turnstile.verify(turnstile_token, websocket.client.host if websocket.client else None):
        await websocket.close(code=4003, reason="Human verification required")
        return None
    
    return key_info


# =============================================================================
# Export all security components
# =============================================================================

__all__ = [
    # Turnstile
    "TurnstileConfig",
    "turnstile",
    "TurnstileMiddleware",
    
    # Rate Limiting
    "limiter",
    "get_rate_limiter",
    "custom_rate_limit_handler",
    "rate_limit",
    "INFERENCE_RATE_LIMIT",
    "MODEL_LOAD_RATE_LIMIT",
    "ADMIN_RATE_LIMIT",
    "HEALTH_RATE_LIMIT",
    "WS_RATE_LIMIT",
    
    # Security Headers
    "SecurityHeadersMiddleware",
    
    # Turnstile Middleware
    "TurnstileMiddleware",
    
    # Request Validation
    "RequestValidationMiddleware",
    
    # Input Validation
    "sanitize_string",
    "validate_eeg_data_shape",
    
    # API Key Management
    "APIKeyManager",
    "api_key_manager",
    "verify_api_key_enhanced",
    "require_scope",
    "require_admin",
    "require_model_management",
    "require_inference",
    "require_admin_stats",
    
    # Configuration
    "SecurityConfig",
    "security_config",
    
    # Middleware Factory
    "create_secure_app",
    
    # WebSocket
    "verify_ws_token",
]