"""
Prediction History API routes.
POST /api/history — save a prediction result (authenticated)
GET  /api/history — retrieve all history for the logged-in user (authenticated)
"""

import json
import os
import shutil

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from psycopg2.extras import RealDictCursor

from app.database import get_connection, release_connection
from app.auth.jwt_utils import get_current_user_payload
from app.history.history_models import SaveHistoryRequest, UpdateHistoryRequest

# Directory where uploaded PDFs are persisted
# Configurable via RESUMES_DIR env var; default is <backend>/data/resumes
_DEFAULT_RESUMES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "resumes")
RESUMES_DIR = os.path.abspath(os.getenv("RESUMES_DIR", _DEFAULT_RESUMES_DIR))
os.makedirs(RESUMES_DIR, exist_ok=True)

router = APIRouter(prefix="/api/history", tags=["History"])


# ──────────────────────────────────────────────────────────────
# POST /api/history  — save a prediction result
# ──────────────────────────────────────────────────────────────

@router.post("", status_code=status.HTTP_201_CREATED)
async def save_history(payload: SaveHistoryRequest, request: Request):
    """
    Save a prediction result to the history table.
    Requires: Authorization: Bearer <token>
    The user_id is extracted from the JWT — users can only save their own records.
    """
    jwt_payload = get_current_user_payload(request)
    user_id = int(jwt_payload["sub"])

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO prediction_history
                    (user_id, prediction_result, input_data, confidence_score,
                     top_predictions, filename)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s)
                RETURNING id, date_created;
                """,
                (
                    user_id,
                    payload.prediction_result,
                    payload.input_data,
                    payload.confidence_score,
                    json.dumps(payload.top_predictions) if payload.top_predictions else None,
                    payload.filename,
                )
            )
            row = dict(cur.fetchone())
            conn.commit()

        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "success": True,
                "id": row["id"],
                "date_created": row["date_created"].isoformat() if row.get("date_created") else None,
                "message": "Prediction saved to history."
            }
        )

    except Exception as e:
        conn.rollback()
        print(f"Save history error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": "Failed to save prediction history."}
        )
    finally:
        release_connection(conn)


# ──────────────────────────────────────────────────────────────
# PATCH /api/history/{id}  — update learning roadmap / certifications
# ──────────────────────────────────────────────────────────────

@router.patch("/{history_id}", status_code=status.HTTP_200_OK)
async def update_history(history_id: int, payload: UpdateHistoryRequest, request: Request):
    """
    Update the learning_roadmap and/or certification_data of an existing
    history record. Only the owner (JWT sub) can update their own records.
    """
    jwt_payload = get_current_user_payload(request)
    user_id = int(jwt_payload["sub"])

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Build dynamic SET clause — only update fields that were provided
            updates = {}
            if payload.learning_roadmap is not None:
                updates["learning_roadmap"] = json.dumps(payload.learning_roadmap)
            if payload.certification_data is not None:
                updates["certification_data"] = json.dumps(payload.certification_data)

            if not updates:
                return JSONResponse(content={"success": True, "message": "Nothing to update."})

            set_clause = ", ".join(f"{k} = %s::jsonb" for k in updates)
            values = list(updates.values()) + [history_id, user_id]

            cur.execute(
                f"""
                UPDATE prediction_history
                SET {set_clause}
                WHERE id = %s AND user_id = %s
                RETURNING id;
                """,
                values
            )
            row = cur.fetchone()
            conn.commit()

        if row is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "Record not found or access denied."}
            )

        return JSONResponse(content={"success": True, "id": row["id"]})

    except Exception as e:
        conn.rollback()
        print(f"Update history error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": "Failed to update history record."}
        )
    finally:
        release_connection(conn)


# ──────────────────────────────────────────────────────────────
# GET /api/history  — get current user's prediction history
# ──────────────────────────────────────────────────────────────

@router.get("")
async def get_history(request: Request):
    """
    Return all prediction history records for the authenticated user.
    Records are returned newest-first.
    Requires: Authorization: Bearer <token>
    """
    jwt_payload = get_current_user_payload(request)
    user_id = int(jwt_payload["sub"])

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, user_id, prediction_result, input_data,
                       confidence_score, top_predictions, filename,
                       learning_roadmap, certification_data, resume_path, date_created
                FROM prediction_history
                WHERE user_id = %s
                ORDER BY date_created DESC;
                """,
                (user_id,)
            )
            rows = cur.fetchall()

        history = []
        for row in rows:
            record = dict(row)
            # Serialize datetime for JSON
            if record.get("date_created"):
                record["date_created"] = record["date_created"].isoformat()
            # confidence_score comes back as Decimal — cast to float
            if record.get("confidence_score") is not None:
                record["confidence_score"] = float(record["confidence_score"])
            # Expose whether a resume PDF is stored (don't leak server path)
            record["has_resume"] = bool(record.pop("resume_path", None))
            history.append(record)

        return JSONResponse(content={
            "success": True,
            "history": history,
            "total": len(history)
        })

    except Exception as e:
        print(f"Get history error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "history": [], "total": 0,
                     "message": "Failed to retrieve prediction history."}
        )
    finally:
        release_connection(conn)


# ──────────────────────────────────────────────────────────────
# POST /api/history/{id}/resume  — attach a PDF to a history record
# ──────────────────────────────────────────────────────────────

@router.post("/{history_id}/resume", status_code=status.HTTP_200_OK)
async def upload_resume(history_id: int, request: Request, file: UploadFile = File(...)):
    """
    Save the uploaded PDF resume for an existing history record.
    The file is stored at <RESUMES_DIR>/<user_id>/<history_id>.pdf and the
    path is persisted in the prediction_history row.
    Requires: Authorization: Bearer <token>
    """
    jwt_payload = get_current_user_payload(request)
    user_id = int(jwt_payload["sub"])

    # Basic validation
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "message": "Only PDF files are accepted."}
        )

    conn = get_connection()
    try:
        # Verify record exists and belongs to this user
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id FROM prediction_history WHERE id = %s AND user_id = %s;",
                (history_id, user_id)
            )
            row = cur.fetchone()

        if row is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"success": False, "message": "Record not found or access denied."}
            )

        # Persist file: <RESUMES_DIR>/<history_id>.pdf
        dest_path = os.path.join(RESUMES_DIR, f"{history_id}.pdf")

        content = await file.read()
        with open(dest_path, "wb") as f:
            f.write(content)

        # Update DB record with path
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE prediction_history
                SET resume_path = %s
                WHERE id = %s AND user_id = %s
                RETURNING id;
                """,
                (dest_path, history_id, user_id)
            )
            conn.commit()

        return JSONResponse(content={"success": True, "message": "Resume stored.", "resume_path": dest_path})

    except Exception as e:
        conn.rollback()
        print(f"Upload resume error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": "Failed to store resume."}
        )
    finally:
        release_connection(conn)


# ──────────────────────────────────────────────────────────────
# GET /api/history/{id}/resume  — download the stored PDF
# ──────────────────────────────────────────────────────────────

@router.get("/{history_id}/resume")
async def download_resume(history_id: int, request: Request):
    """
    Stream the stored PDF resume for a history record back to the caller.
    Only the owner (JWT sub) can access their own resume.
    Requires: Authorization: Bearer <token>
    """
    jwt_payload = get_current_user_payload(request)
    user_id = int(jwt_payload["sub"])

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT resume_path, filename FROM prediction_history WHERE id = %s AND user_id = %s;",
                (history_id, user_id)
            )
            row = cur.fetchone()
    finally:
        release_connection(conn)

    if row is None:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "message": "Record not found or access denied."}
        )

    resume_path = row.get("resume_path")
    if not resume_path or not os.path.exists(resume_path):
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "message": "Resume file not found."}
        )

    original_name = row.get("filename") or f"resume_{history_id}.pdf"
    return FileResponse(
        path=resume_path,
        media_type="application/pdf",
        filename=original_name,
    )
