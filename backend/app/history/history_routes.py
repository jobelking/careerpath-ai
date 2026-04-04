"""
Prediction History API routes.
POST /api/history — save a prediction result (authenticated)
GET  /api/history — retrieve all history for the logged-in user (authenticated)
"""

import json

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import JSONResponse
from psycopg2.extras import RealDictCursor

from app.database import get_connection, release_connection
from app.auth.jwt_utils import get_current_user_payload
from app.history.history_models import SaveHistoryRequest, UpdateHistoryRequest
from app.services.storage_service import upload_resume, get_resume_signed_url

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
            # PostgreSQL rejects NUL bytes (0x00) — strip them from text fields
            safe_input_data = payload.input_data.replace("\x00", "") if payload.input_data else None
            cur.execute(
                """
                INSERT INTO prediction_history
                    (user_id, prediction_result, input_data, confidence_score,
                     top_predictions, filename, extracted_keywords,
                     extracted_keywords_by_path, total_distinctive_keywords,
                     total_distinctive_keywords_by_path)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb, %s, %s::jsonb)
                RETURNING id, date_created;
                """,
                (
                    user_id,
                    payload.prediction_result,
                    safe_input_data,
                    payload.confidence_score,
                    json.dumps(payload.top_predictions) if payload.top_predictions else None,
                    payload.filename,
                    json.dumps(payload.extracted_keywords) if payload.extracted_keywords else None,
                    json.dumps(payload.extracted_keywords_by_path) if payload.extracted_keywords_by_path else None,
                    payload.total_distinctive_keywords,
                    json.dumps(payload.total_distinctive_keywords_by_path) if payload.total_distinctive_keywords_by_path else None,
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
            if payload.learning_roadmap_by_path is not None:
                updates["learning_roadmap_by_path"] = json.dumps(payload.learning_roadmap_by_path)
            if payload.certification_data_by_path is not None:
                updates["certification_data_by_path"] = json.dumps(payload.certification_data_by_path)
            if payload.skills_insights_by_path is not None:
                updates["skills_insights_by_path"] = json.dumps(payload.skills_insights_by_path)

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
                      extracted_keywords, extracted_keywords_by_path,
                      total_distinctive_keywords, total_distinctive_keywords_by_path,
                      learning_roadmap, certification_data,
                      learning_roadmap_by_path, certification_data_by_path,
                      skills_insights_by_path,
                      resume_path, date_created
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
                record["date_created"] = record["date_created"].isoformat() + "Z"
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
async def upload_resume_endpoint(history_id: int, request: Request, file: UploadFile = File(...)):
    """
    Upload the resume PDF to Supabase Storage and store the object path
    in the prediction_history row.
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

        # Upload to Supabase Storage
        content = await file.read()
        object_path = upload_resume(user_id, history_id, content)

        # Store the Supabase Storage object path in the DB
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE prediction_history
                SET resume_path = %s
                WHERE id = %s AND user_id = %s
                RETURNING id;
                """,
                (object_path, history_id, user_id)
            )
            conn.commit()

        return JSONResponse(content={"success": True, "message": "Resume uploaded to storage."})

    except Exception as e:
        conn.rollback()
        print(f"Upload resume error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": "Failed to upload resume."}
        )
    finally:
        release_connection(conn)


# ──────────────────────────────────────────────────────────────
# GET /api/history/{id}/resume  — get a signed URL for the stored PDF
# ──────────────────────────────────────────────────────────────

@router.get("/{history_id}/resume")
async def download_resume(history_id: int, request: Request):
    """
    Return a short-lived signed URL for the resume stored in Supabase Storage.
    The frontend should redirect the user's browser to this URL to download the PDF.
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
    if not resume_path:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"success": False, "message": "No resume stored for this record."}
        )

    try:
        signed_url = get_resume_signed_url(resume_path)
    except Exception as e:
        print(f"Signed URL error: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": "Failed to generate resume download link."}
        )

    return JSONResponse(content={
        "success": True,
        "url": signed_url,
        "filename": row.get("filename") or f"resume_{history_id}.pdf"
    })
