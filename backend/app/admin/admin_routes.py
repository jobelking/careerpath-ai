"""
Admin API routes — full CRUD for users and prediction history.

All endpoints:
  - Require a valid JWT (Authorization: Bearer <token>)
  - Require is_admin=True in the JWT payload
  - Use parameterized SQL to prevent injection
  - Return consistent JSON envelopes

Prefix: /api/admin
"""

import json
import os
import bcrypt
from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from psycopg2.extras import RealDictCursor

from app.database import get_connection, release_connection
from app.auth.jwt_utils import get_current_user_payload
from app.admin.admin_models import (
    AdminCreateUserRequest,
    AdminUpdateUserRequest,
    AdminUserRecord,
    AdminUsersResponse,
    AdminHistoryRecord,
    AdminHistoryResponse,
    AdminStatsResponse,
)

router = APIRouter(prefix="/api/admin", tags=["Admin"])


# ─────────────────────────────────────────────────────────────────────────────
# Helper: enforce admin access
# ─────────────────────────────────────────────────────────────────────────────

def require_admin(request: Request) -> dict:
    """
    Extract and verify the JWT from the request, then assert is_admin=True.
    Returns the decoded JWT payload on success.
    Raises 401 / 403 on failure.
    """
    payload = get_current_user_payload(request)
    if not payload.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required."
        )
    return payload


# ═════════════════════════════════════════════════════════════════════════════
# STATS
# ═════════════════════════════════════════════════════════════════════════════

@router.get("/stats", response_model=AdminStatsResponse)
async def get_stats(request: Request):
    """
    Return high-level counts for the admin dashboard overview.
    - total_users, verified_users, admin_users, total_predictions
    """
    require_admin(request)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT
                    COUNT(*)                          AS total_users,
                    COUNT(*) FILTER (WHERE is_verified) AS verified_users,
                    COUNT(*) FILTER (WHERE is_admin)    AS admin_users
                FROM users;
            """)
            user_stats = dict(cur.fetchone())

            cur.execute("SELECT COUNT(*) AS total_predictions FROM prediction_history;")
            pred_stats = dict(cur.fetchone())

        return AdminStatsResponse(
            total_users=user_stats["total_users"],
            verified_users=user_stats["verified_users"],
            admin_users=user_stats["admin_users"],
            total_predictions=pred_stats["total_predictions"],
        )
    finally:
        release_connection(conn)


# ═════════════════════════════════════════════════════════════════════════════
# USER CRUD
# ═════════════════════════════════════════════════════════════════════════════

# ── GET /api/admin/users ──────────────────────────────────────────────────────

@router.get("/users", response_model=AdminUsersResponse)
async def list_users(
    request: Request,
    search: str = "",
    verified: str = "all",   # "all" | "true" | "false"
    is_admin: str = "all",   # "all" | "true" | "false"
    page: int = 1,
    page_size: int = 20,
):
    """
    List all users with optional search/filter and pagination.
    - search: partial match on username or email (case-insensitive)
    - verified: filter by is_verified flag
    - is_admin: filter by is_admin flag
    - page / page_size: pagination
    """
    require_admin(request)

    # Build dynamic WHERE clauses safely using parameterised values
    conditions = []
    params: list = []

    if search:
        conditions.append("(LOWER(username) LIKE %s OR LOWER(email) LIKE %s)")
        like = f"%{search.lower()}%"
        params += [like, like]

    if verified == "true":
        conditions.append("is_verified = TRUE")
    elif verified == "false":
        conditions.append("is_verified = FALSE")

    if is_admin == "true":
        conditions.append("is_admin = TRUE")
    elif is_admin == "false":
        conditions.append("is_admin = FALSE")

    where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    offset = (page - 1) * page_size

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Total count for pagination metadata
            cur.execute(
                f"SELECT COUNT(*) AS total FROM users {where_sql};",
                params
            )
            total = cur.fetchone()["total"]

            # Paginated result set
            cur.execute(
                f"""
                SELECT id, username, email, is_verified, is_admin, created_at
                FROM users {where_sql}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s;
                """,
                params + [page_size, offset]
            )
            rows = [AdminUserRecord(**dict(r)) for r in cur.fetchall()]

        return AdminUsersResponse(success=True, users=rows, total=total)
    finally:
        release_connection(conn)


# ── GET /api/admin/users/{user_id} ───────────────────────────────────────────

@router.get("/users/{user_id}", response_model=AdminUserRecord)
async def get_user(user_id: int, request: Request):
    """Return a single user by ID."""
    require_admin(request)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, username, email, is_verified, is_admin, created_at FROM users WHERE id = %s;",
                (user_id,)
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="User not found.")
        return AdminUserRecord(**dict(row))
    finally:
        release_connection(conn)


# ── POST /api/admin/users ─────────────────────────────────────────────────────

@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(payload: AdminCreateUserRequest, request: Request):
    """
    Create a new user from the admin dashboard.
    The account is auto-verified (no OTP required for admin-created accounts).
    Password is hashed with bcrypt (work factor 12).
    """
    require_admin(request)
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Prevent duplicate email
            cur.execute("SELECT id FROM users WHERE email = %s;", (payload.email.lower(),))
            if cur.fetchone():
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A user with this email already exists."
                )

            # Hash password with bcrypt
            password_hash = bcrypt.hashpw(
                payload.password.encode("utf-8"),
                bcrypt.gensalt(rounds=12)
            ).decode("utf-8")

            cur.execute(
                """
                INSERT INTO users (username, email, password_hash, is_verified, is_admin)
                VALUES (%s, %s, %s, TRUE, %s)
                RETURNING id, username, email, is_verified, is_admin, created_at;
                """,
                (payload.username.strip(), payload.email.lower(), password_hash, payload.is_admin)
            )
            new_user = dict(cur.fetchone())
            conn.commit()

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={"success": True, "user": {
                "id": new_user["id"],
                "username": new_user["username"],
                "email": new_user["email"],
                "is_verified": new_user["is_verified"],
                "is_admin": new_user["is_admin"],
                "created_at": new_user["created_at"].isoformat() if new_user.get("created_at") else None,
            }}
        )

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        print(f"Admin create user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user.")
    finally:
        release_connection(conn)


# ── PUT /api/admin/users/{user_id} ───────────────────────────────────────────

@router.put("/users/{user_id}")
async def update_user(user_id: int, payload: AdminUpdateUserRequest, request: Request):
    """
    Update a user's profile fields.
    Only the provided (non-None) fields are written.
    If a new password is provided it will be re-hashed.
    """
    require_admin(request)

    # Build SET clause dynamically from non-None fields
    set_parts = []
    params: list = []

    if payload.username is not None:
        set_parts.append("username = %s")
        params.append(payload.username.strip())

    if payload.email is not None:
        set_parts.append("email = %s")
        params.append(payload.email.lower())

    if payload.is_admin is not None:
        set_parts.append("is_admin = %s")
        params.append(payload.is_admin)

    if payload.is_verified is not None:
        set_parts.append("is_verified = %s")
        params.append(payload.is_verified)

    if payload.password is not None:
        hashed = bcrypt.hashpw(
            payload.password.encode("utf-8"), bcrypt.gensalt(rounds=12)
        ).decode("utf-8")
        set_parts.append("password_hash = %s")
        params.append(hashed)

    if not set_parts:
        raise HTTPException(status_code=400, detail="No fields provided for update.")

    params.append(user_id)  # for the WHERE clause

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                UPDATE users SET {', '.join(set_parts)}
                WHERE id = %s
                RETURNING id, username, email, is_verified, is_admin, created_at;
                """,
                params
            )
            updated = cur.fetchone()
            if not updated:
                raise HTTPException(status_code=404, detail="User not found.")
            conn.commit()

        updated = dict(updated)
        return {
            "success": True,
            "user": {
                "id": updated["id"],
                "username": updated["username"],
                "email": updated["email"],
                "is_verified": updated["is_verified"],
                "is_admin": updated["is_admin"],
                "created_at": updated["created_at"].isoformat() if updated.get("created_at") else None,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        print(f"Admin update user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to update user.")
    finally:
        release_connection(conn)


# ── DELETE /api/admin/users/{user_id} ────────────────────────────────────────

@router.delete("/users/{user_id}", status_code=status.HTTP_200_OK)
async def delete_user(user_id: int, request: Request):
    """
    Delete a user and all their associated prediction history (CASCADE).
    An admin cannot delete their own account via this endpoint.
    """
    jwt_payload = require_admin(request)
    calling_admin_id = int(jwt_payload["sub"])

    if calling_admin_id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot delete your own admin account."
        )

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s RETURNING id;", (user_id,))
            deleted = cur.fetchone()
            if not deleted:
                raise HTTPException(status_code=404, detail="User not found.")
            conn.commit()

        return {"success": True, "message": f"User {user_id} deleted."}

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        print(f"Admin delete user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete user.")
    finally:
        release_connection(conn)


# ═════════════════════════════════════════════════════════════════════════════
# PREDICTION HISTORY CRUD
# ═════════════════════════════════════════════════════════════════════════════

# ── GET /api/admin/history ────────────────────────────────────────────────────

@router.get("/history", response_model=AdminHistoryResponse)
async def list_history(
    request: Request,
    search: str = "",       # partial match on prediction_result or filename
    user_id: int = 0,       # 0 = all users
    page: int = 1,
    page_size: int = 20,
):
    """
    List all prediction history records across all users.
    - search: partial match on prediction_result or filename
    - user_id: restrict to a specific user (0 = all)
    - page / page_size: pagination
    """
    require_admin(request)

    conditions = []
    params: list = []

    if search:
        conditions.append(
            "(LOWER(ph.prediction_result) LIKE %s OR LOWER(ph.filename) LIKE %s)"
        )
        like = f"%{search.lower()}%"
        params += [like, like]

    if user_id:
        conditions.append("ph.user_id = %s")
        params.append(user_id)

    where_sql = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    offset = (page - 1) * page_size

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                f"""
                SELECT COUNT(*) AS total
                FROM prediction_history ph
                JOIN users u ON u.id = ph.user_id
                {where_sql};
                """,
                params
            )
            total = cur.fetchone()["total"]

            cur.execute(
                f"""
                SELECT
                    ph.id, ph.user_id, u.email AS user_email, u.username,
                    ph.prediction_result, ph.confidence_score,
                    ph.filename, ph.date_created, ph.top_predictions,
                    (ph.resume_path IS NOT NULL) AS has_resume
                FROM prediction_history ph
                JOIN users u ON u.id = ph.user_id
                {where_sql}
                ORDER BY ph.date_created DESC
                LIMIT %s OFFSET %s;
                """,
                params + [page_size, offset]
            )
            rows = []
            for r in cur.fetchall():
                row_dict = dict(r)
                # top_predictions may already be a list (psycopg2 JSONB auto-decodes)
                if isinstance(row_dict.get("top_predictions"), str):
                    try:
                        row_dict["top_predictions"] = json.loads(row_dict["top_predictions"])
                    except Exception:
                        row_dict["top_predictions"] = None
                rows.append(AdminHistoryRecord(**row_dict))

        return AdminHistoryResponse(success=True, history=rows, total=total)
    finally:
        release_connection(conn)


# ── GET /api/admin/history/{id}/resume ───────────────────────────────────────

@router.get("/history/{record_id}/resume")
async def admin_download_resume(record_id: int, request: Request):
    """
    Stream the stored PDF resume for any prediction history record.
    Requires admin privileges.
    """
    require_admin(request)

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT resume_path, filename FROM prediction_history WHERE id = %s;",
                (record_id,)
            )
            row = cur.fetchone()
    finally:
        release_connection(conn)

    if row is None:
        raise HTTPException(status_code=404, detail="History record not found.")

    resume_path = row.get("resume_path")
    if not resume_path or not os.path.exists(resume_path):
        raise HTTPException(status_code=404, detail="Resume file not stored for this record.")

    original_name = row.get("filename") or f"resume_{record_id}.pdf"
    return FileResponse(
        path=resume_path,
        media_type="application/pdf",
        filename=original_name,
    )


# ── DELETE /api/admin/history/{record_id} ────────────────────────────────────

@router.delete("/history/{record_id}", status_code=status.HTTP_200_OK)
async def delete_history_record(record_id: int, request: Request):
    """Delete a specific prediction history record by ID."""
    require_admin(request)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM prediction_history WHERE id = %s RETURNING id;",
                (record_id,)
            )
            deleted = cur.fetchone()
            if not deleted:
                raise HTTPException(status_code=404, detail="History record not found.")
            conn.commit()

        return {"success": True, "message": f"History record {record_id} deleted."}

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        print(f"Admin delete history error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete history record.")
    finally:
        release_connection(conn)
