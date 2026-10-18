"""
NeuroMove Enterprise Backend - Security/Authentication
Keycloak OIDC Integration
"""
from functools import lru_cache
from typing import Optional, List, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from datetime import datetime, timedelta
import httpx
import json
from app.core.config import settings


security = HTTPBearer(auto_error=False)


class KeycloakClient:
    """Keycloak OIDC client for authentication."""

    def __init__(self):
        self.server_url = settings.KEYCLOAK_URL.rstrip("/")
        self.realm = settings.KEYCLOAK_REALM
        self.client_id = settings.KEYCLOAK_CLIENT_ID
        self.client_secret = settings.KEYCLOAK_CLIENT_SECRET
        self._jwks_cache: Optional[Dict] = None
        self._jwks_cache_time: float = 0

    @property
    def jwks_url(self) -> str:
        return f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/certs"

    @property
    def issuer_url(self) -> str:
        return f"{self.server_url}/realms/{self.realm}"

    @property
    def token_url(self) -> str:
        return f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/token"

    @property
    def userinfo_url(self) -> str:
        return f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/userinfo"

    @property
    def logout_url(self) -> str:
        return f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/logout"

    async def get_jwks(self) -> Dict:
        """Fetch and cache JWKS from Keycloak."""
        import time
        now = time.time()
        if self._jwks_cache and (now - self._jwks_cache_time) < 3600:
            return self._jwks_cache
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(self.jwks_url)
            response.raise_for_status()
            self._jwks_cache = response.json()
            self._jwks_cache_time = now
            return self._jwks_cache

    async def get_public_key(self, kid: str):
        jwks = await self.get_jwks()
        from jose.algorithms import RSAAlgorithm
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                return RSAAlgorithm.from_jwk(json.dumps(key))
        return None

    async def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT access token from Keycloak."""
        try:
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            if not kid:
                raise HTTPException(status_code=401, detail="Token missing kid header")

            public_key = await self.get_public_key(kid)
            if not public_key:
                raise HTTPException(status_code=401, detail="Unable to find matching key")

            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                issuer=f"{self.server_url}/realms/{self.realm}",
                options={"verify_aud": False},  # Keycloak uses "account"
            )
            return payload
        except JWTError as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

    async def introspect_token(self, token: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/token/introspect",
                data={
                    "token": token,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            return response.json()

    def get_login_url(self, redirect_uri: str, state: Optional[str] = None) -> str:
        """Generate Keycloak login URL."""
        from urllib.parse import urlencode
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile roles",
        }
        if state:
            params["state"] = state
        return f"{self.server_url}/realms/{self.realm}/protocol/openid-connect/auth?{params}"

    def get_logout_url(self, redirect_uri: Optional[str] = None) -> str:
        params = {}
        if redirect_uri:
            params["post_logout_redirect_uri"] = redirect_uri
        return f"{self.logout_url}?{params}" if params else self.logout_url

    async def exchange_code_for_token(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                self.token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            return response.json()

    async def get_userinfo(self, access_token: str) -> Dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                self.userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"},
            )
            response.raise_for_status()
            return response.json()


# Global Keycloak client instance
keycloak = KeycloakClient()


# FastAPI Dependencies
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """Verify JWT token and return user info."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        payload = await keycloak.verify_token(credentials.credentials)
        return {
            "sub": payload.get("sub"),
            "email": payload.get("email"),
            "preferred_username": payload.get("preferred_username"),
            "name": payload.get("name"),
            "roles": payload.get("realm_access", {}).get("roles", []),
            "groups": payload.get("groups", []),
            "email_verified": payload.get("email_verified", False),
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_roles(*allowed_roles: str):
    """Dependency to require specific user roles."""
    async def role_checker(user: Dict = Depends(get_current_user)) -> Dict:
        user_roles = set(user.get("roles", []))
        if not any(role in user_roles for role in allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(allowed_roles)}",
            )
        return user
    return role_checker


# Convenience dependencies
require_admin = require_roles("admin")
require_researcher = require_roles("admin", "researcher")
require_clinician = require_roles("admin", "researcher", "clinician")


# Local token fallback (for demo/dev without Keycloak)
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES


def create_local_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def verify_local_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


USE_KEYCLOAK = bool(settings.KEYCLOAK_URL and settings.KEYCLOAK_URL != "http://localhost:8080")


async def get_current_user_flexible(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Dict[str, Any]:
    """Flexible auth: tries Keycloak first, falls back to local JWT."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if USE_KEYCLOAK:
        return await keycloak.verify_token(credentials.credentials)
    else:
        return await verify_local_token(credentials.credentials)