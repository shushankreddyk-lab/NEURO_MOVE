"""
Authentication endpoints
"""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from app.core.security import (
    create_local_access_token,
    verify_local_token,
    keycloak,
    USE_KEYCLOAK,
    security,
    security as http_security,
)
from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.models.base import User
from app.schemas.base import LoginRequest, UserResponse, Token
from sqlalchemy import select
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

router = APIRouter()


@router.post("/login", response_model=Token)
async def login(request: LoginRequest):
    """Authenticate user via Keycloak or local fallback."""
    if USE_KEYCLOAK:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{settings.KEYCLOAK_URL}/realms/{settings.KEYCLOAK_REALM}/protocol/openid-connect/token",
                    data={
                        "grant_type": "password",
                        "username": request.username,
                        "password": request.password,
                        "client_id": settings.KEYCLOAK_CLIENT_ID,
                        "client_secret": settings.KEYCLOAK_CLIENT_SECRET,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return Token(access_token=data["access_token"], token_type="bearer")
        except Exception:
            pass

    async with AsyncSessionLocal() as db:
        result = await db.execute(select(User).where(User.username == request.username))
        user = result.scalar_one_or_none()
        if not user or not pwd_context.verify(request.password, user.hashed_password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        if not user.is_active:
            raise HTTPException(status_code=401, detail="User inactive")

    access_token = create_local_access_token(data={"sub": user.username})
    return Token(access_token=access_token)


@router.post("/register", response_model=UserResponse)
async def register(request: LoginRequest):
    async with AsyncSessionLocal() as db:
        existing = await db.execute(select(User).where(User.username == request.username))
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=400, detail="Username already registered")

        user = User(
            username=request.username,
            email=request.email,
            hashed_password=pwd_context.hash(request.password),
            role="clinician",
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if not credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")

    if USE_KEYCLOAK:
        payload = await keycloak.verify_token(credentials.credentials)
    else:
        payload = await verify_local_token(credentials.credentials)

    async with AsyncSessionLocal() as db:
        username = payload.get("sub", payload.get("preferred_username", "unknown"))
        result = await db.execute(select(User).where(User.username == username))
        user = result.scalar_one_or_none()
        if not user:
            user = User(
                username=username,
                email=payload.get("email", ""),
                role="clinician",
                keycloak_id=payload.get("sub"),
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        return user