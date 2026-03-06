"""
JWT utility functions for CareerPath AI authentication.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, Request
from dotenv import load_dotenv

load_dotenv()

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY environment variable is not set. Add it to your .env file.")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))


def create_access_token(user_id: int, email: str, is_admin: bool = False) -> str:
    """
    Create a signed JWT token containing the user id, email, and admin flag.
    Expires after JWT_EXPIRY_HOURS hours (default 24).
    """
    payload = {
        "sub": str(user_id),
        "email": email,
        "is_admin": is_admin,
        "exp": datetime.now(timezone.utc) + timedelta(hours=EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    """
    Decode and verify a JWT token.
    Raises HTTPException 401 if invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token. Please log in again."
        )


def get_token_from_request(request: Request) -> str:
    """
    Extract Bearer token from the Authorization header.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Authorization header missing or malformed. Expected 'Bearer <token>'."
        )
    return auth_header.split(" ", 1)[1]


def get_current_user_payload(request: Request) -> dict:
    """
    Full helper: extract token from request + verify it.
    Returns the decoded JWT payload.
    """
    token = get_token_from_request(request)
    return verify_token(token)
