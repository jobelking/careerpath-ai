"""
Pydantic models for Admin API request/response validation.
"""

from pydantic import BaseModel, EmailStr, field_validator
from datetime import datetime
from typing import Optional, List, Any


# ── User management models ─────────────────────────────────────────────────────

class AdminCreateUserRequest(BaseModel):
    """Payload to create a new user from the admin dashboard."""
    username: str
    email: EmailStr
    password: str
    is_admin: bool = False

    @field_validator("username")
    @classmethod
    def username_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Username cannot be empty")
        if len(v) > 100:
            raise ValueError("Username must be 100 characters or fewer")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class AdminUpdateUserRequest(BaseModel):
    """Partial update — only provided fields are written."""
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    is_admin: Optional[bool] = None
    is_verified: Optional[bool] = None
    password: Optional[str] = None  # if set, will be re-hashed

    @field_validator("username")
    @classmethod
    def username_not_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.strip()
            if not v:
                raise ValueError("Username cannot be empty")
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v) < 6:
            raise ValueError("Password must be at least 6 characters")
        return v


class AdminUserRecord(BaseModel):
    """User row returned by admin list/detail endpoints."""
    id: int
    username: str
    email: str
    is_verified: bool
    is_admin: bool
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class AdminUsersResponse(BaseModel):
    success: bool
    users: List[AdminUserRecord]
    total: int


# ── History management models ──────────────────────────────────────────────────

class AdminHistoryRecord(BaseModel):
    """Prediction history row with the owner's email included."""
    id: int
    user_id: int
    user_email: Optional[str] = None
    username: Optional[str] = None
    prediction_result: str
    confidence_score: Optional[float] = None
    filename: Optional[str] = None
    date_created: Optional[datetime] = None
    top_predictions: Optional[List[Any]] = None

    class Config:
        from_attributes = True


class AdminHistoryResponse(BaseModel):
    success: bool
    history: List[AdminHistoryRecord]
    total: int


# ── Stats model ────────────────────────────────────────────────────────────────

class AdminStatsResponse(BaseModel):
    total_users: int
    verified_users: int
    admin_users: int
    total_predictions: int
