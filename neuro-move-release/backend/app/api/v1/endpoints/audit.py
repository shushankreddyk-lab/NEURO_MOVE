"""
Audit log endpoints
"""
from fastapi import APIRouter, Depends
from app.core.security import get_current_user, require_admin
from app.db.database import AsyncSessionLocal
from app.models.base import AuditLog
from app.schemas.base import AuditLogResponse
from sqlalchemy import select

router = APIRouter()


@router.get("", response_model=List[AuditLogResponse])
async def list_audit_logs(current_user: dict = Depends(require_admin)):
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(100))
        return result.scalars().all()