"""
OTP utility functions for secure 6-digit code generation and expiry.
Uses secrets module for cryptographically secure randomness.
"""

import secrets
from datetime import datetime, timedelta, timezone

OTP_EXPIRY_MINUTES = 5
MAX_OTP_ATTEMPTS = 5
RESEND_COOLDOWN_SECONDS = 60


def generate_otp() -> str:
    """Generate a cryptographically secure 6-digit OTP code."""
    # secrets.randbelow is CSPRNG-backed — safe for security purposes
    return f"{secrets.randbelow(1_000_000):06d}"


def get_expiration() -> datetime:
    """Return UTC timestamp 10 minutes from now."""
    return datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRY_MINUTES)


def is_expired(expiration: datetime) -> bool:
    """Check whether the OTP expiration timestamp has passed."""
    if expiration is None:
        return True
    # Ensure both datetimes are timezone-aware for comparison
    if expiration.tzinfo is None:
        expiration = expiration.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) > expiration


def is_resend_allowed(last_sent_at: datetime) -> bool:
    """Return True if the cooldown period has elapsed since last send."""
    if last_sent_at is None:
        return True
    if last_sent_at.tzinfo is None:
        last_sent_at = last_sent_at.replace(tzinfo=timezone.utc)
    elapsed = (datetime.now(timezone.utc) - last_sent_at).total_seconds()
    return elapsed >= RESEND_COOLDOWN_SECONDS


def seconds_until_resend_allowed(last_sent_at: datetime) -> int:
    """Return the number of seconds remaining in the resend cooldown."""
    if last_sent_at is None:
        return 0
    if last_sent_at.tzinfo is None:
        last_sent_at = last_sent_at.replace(tzinfo=timezone.utc)
    elapsed = (datetime.now(timezone.utc) - last_sent_at).total_seconds()
    remaining = RESEND_COOLDOWN_SECONDS - elapsed
    return max(0, int(remaining))
