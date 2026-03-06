"""
Prediction History API routes.
POST /api/history — save a prediction result (authenticated)
GET  /api/history — retrieve all history for the logged-in user (authenticated)
"""

import json
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from psycopg2.extras import RealDictCursor

from app.database import get_connection, release_connection
from app.auth.jwt_utils import get_current_user_payload
from app.history.history_models import SaveHistoryRequest, UpdateHistoryRequest

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
                       learning_roadmap, certification_data, date_created
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
