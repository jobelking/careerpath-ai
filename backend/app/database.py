"""
PostgreSQL Database Connection and Setup for CareerPath AI
Handles connection pooling and auto-creates the users and prediction_history tables.
"""

import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Connection pool (min 1, max 10 connections)
_connection_pool: pool.SimpleConnectionPool | None = None


def get_pool() -> pool.SimpleConnectionPool:
    """Initialize or return the existing connection pool."""
    global _connection_pool
    if _connection_pool is None:
        try:
            _connection_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=DATABASE_URL,
            )
            print("✅ PostgreSQL connection pool created successfully.")
        except psycopg2.OperationalError as e:
            print(f"❌ Failed to connect to PostgreSQL: {e}")
            raise
    return _connection_pool


def get_connection():
    """Get a connection from the pool. Raises 503 if DB is unavailable."""
    from fastapi import HTTPException
    try:
        return get_pool().getconn()
    except Exception:
        raise HTTPException(
            status_code=503,
            detail="Database is unavailable. Please ensure PostgreSQL is running."
        )


def release_connection(conn):
    """Return a connection to the pool."""
    get_pool().putconn(conn)


def init_db():
    """
    Create database tables if they don't exist.
    Uses parameterized queries — safe from SQL injection.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id                SERIAL PRIMARY KEY,
                    username          VARCHAR(100) NOT NULL,
                    email             VARCHAR(255) UNIQUE NOT NULL,
                    password_hash     VARCHAR(255) NOT NULL,
                    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    verification_code VARCHAR(6),
                    code_expiration   TIMESTAMP WITH TIME ZONE,
                    is_verified       BOOLEAN NOT NULL DEFAULT FALSE,
                    otp_attempts      INT NOT NULL DEFAULT 0,
                    last_otp_sent_at  TIMESTAMP WITH TIME ZONE
                );
            """)
            # Migrate existing table: rename 'name' → 'username' if old column exists
            cur.execute("""
                DO $$
                BEGIN
                    IF EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='users' AND column_name='name'
                    ) THEN
                        ALTER TABLE users RENAME COLUMN name TO username;
                    END IF;
                END$$;
            """)
            # Add OTP columns if they don't exist yet (safe migration for existing tables)
            otp_columns = [
                ("verification_code", "VARCHAR(6)"),
                ("code_expiration",   "TIMESTAMP WITH TIME ZONE"),
                ("is_verified",       "BOOLEAN NOT NULL DEFAULT FALSE"),
                ("otp_attempts",      "INT NOT NULL DEFAULT 0"),
                ("last_otp_sent_at",  "TIMESTAMP WITH TIME ZONE"),
                # Admin flag — first registered user can be promoted via DB or seed script
                ("is_admin",          "BOOLEAN NOT NULL DEFAULT FALSE"),
            ]
            for col_name, col_type in otp_columns:
                cur.execute(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='users' AND column_name='{col_name}'
                        ) THEN
                            ALTER TABLE users ADD COLUMN {col_name} {col_type};
                        END IF;
                    END$$;
                """)

            # ── prediction_history table ──────────────────────────────────────────
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id                SERIAL PRIMARY KEY,
                    user_id           INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    prediction_result VARCHAR(255) NOT NULL,
                    input_data        TEXT,
                    confidence_score  NUMERIC(5,2),
                    top_predictions   JSONB,
                    filename          VARCHAR(255),
                    extracted_keywords JSONB,
                    total_distinctive_keywords INTEGER,
                    learning_roadmap  JSONB,
                    certification_data JSONB,
                    resume_path       TEXT,
                    date_created      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Migrate existing prediction_history: add new columns if missing
            history_columns = [
                ("learning_roadmap",    "JSONB"),
                ("certification_data",  "JSONB"),
                ("resume_path",         "TEXT"),
                ("extracted_keywords",  "JSONB"),
                ("total_distinctive_keywords",  "INTEGER"),
            ]
            for col_name, col_type in history_columns:
                cur.execute(f"""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1 FROM information_schema.columns
                            WHERE table_name='prediction_history' AND column_name='{col_name}'
                        ) THEN
                            ALTER TABLE prediction_history ADD COLUMN {col_name} {col_type};
                        END IF;
                    END$$;
                """)
            # Removed the stray closing triple-quote here.

            conn.commit()
            print("✅ Database tables verified/created (users + prediction_history).")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error initializing database: {e}")
        raise
    finally:
        release_connection(conn)

