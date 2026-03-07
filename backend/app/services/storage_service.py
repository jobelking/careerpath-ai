"""
Supabase Storage service for resume PDF management.
Handles upload, signed URL generation, and deletion of resume files
stored in the 'resumes' Supabase Storage bucket.
"""

import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
BUCKET_NAME = "resumes"

_client: Client | None = None


def get_supabase() -> Client:
    """Return (or lazily initialise) the Supabase client."""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment variables."
            )
        _client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _client


def upload_resume(user_id: int, history_id: int, file_bytes: bytes) -> str:
    """
    Upload a resume PDF to Supabase Storage.

    Args:
        user_id:    Owner's user ID (used for path namespacing)
        history_id: Prediction history record ID
        file_bytes: Raw PDF bytes

    Returns:
        object_path: Storage path, e.g. "resumes/<user_id>/<history_id>.pdf"
    """
    object_path = f"{user_id}/{history_id}.pdf"
    supabase = get_supabase()

    # Remove any existing file first (upsert-style)
    try:
        supabase.storage.from_(BUCKET_NAME).remove([object_path])
    except Exception:
        pass  # Ignore if it doesn't exist yet

    supabase.storage.from_(BUCKET_NAME).upload(
        path=object_path,
        file=file_bytes,
        file_options={"content-type": "application/pdf", "upsert": "true"},
    )
    return object_path


def get_resume_signed_url(object_path: str, expires_in: int = 3600) -> str:
    """
    Generate a signed download URL for a stored resume.

    Args:
        object_path: Storage path as returned by upload_resume()
        expires_in:  URL validity in seconds (default 1 hour)

    Returns:
        Signed URL string
    """
    supabase = get_supabase()
    response = supabase.storage.from_(BUCKET_NAME).create_signed_url(
        object_path, expires_in
    )
    signed_url = response.get("signedURL") or response.get("signed_url") or response.get("signedUrl")
    if not signed_url:
        raise RuntimeError(f"Failed to generate signed URL for {object_path}: {response}")
    return signed_url


def delete_resume(object_path: str) -> None:
    """
    Delete a resume file from Supabase Storage.

    Args:
        object_path: Storage path as returned by upload_resume()
    """
    try:
        supabase = get_supabase()
        supabase.storage.from_(BUCKET_NAME).remove([object_path])
    except Exception as e:
        # Log but don't raise — a missing file shouldn't break DB deletions
        print(f"⚠️  Warning: could not delete resume from storage ({object_path}): {e}")
