"""
Authentication API routes — register, login, verify OTP, resend OTP, and get current user.
All DB queries use parameterized statements to prevent SQL injection.
"""

import bcrypt
from fastapi import APIRouter, HTTPException, Request, status
from psycopg2.extras import RealDictCursor

from app.database import get_connection, release_connection
from app.auth.auth_models import (
    RegisterRequest, LoginRequest, AuthResponse,
    UserResponse, VerifyOTPRequest, ResendOTPRequest
)
from app.auth.jwt_utils import create_access_token, get_current_user_payload
from app.auth.otp_utils import (
    generate_otp, get_expiration, is_expired,
    is_resend_allowed, seconds_until_resend_allowed,
    MAX_OTP_ATTEMPTS
)
from app.services.email_service import send_verification_email

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# ─────────────────────────────────────────────
# POST /api/auth/register
# ─────────────────────────────────────────────

@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest):
    """
    Register a new user.
    - Validates input via Pydantic
    - Checks for duplicate email
    - Hashes password with bcrypt (work factor 12)
    - Generates a 6-digit OTP and sends it to the user's email
    - Returns requires_verification=True (no JWT until email is confirmed)
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Check for existing email
            cur.execute(
                "SELECT id, is_verified FROM users WHERE email = %s;",
                (payload.email.lower(),)
            )
            existing = cur.fetchone()
            if existing:
                if existing["is_verified"]:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="An account with this email address already exists."
                    )
                else:
                    # Account exists but not verified — resend OTP
                    otp = generate_otp()
                    expiration = get_expiration()
                    cur.execute(
                        """
                        UPDATE users
                        SET verification_code = %s,
                            code_expiration   = %s,
                            otp_attempts      = 0,
                            last_otp_sent_at  = NOW() AT TIME ZONE 'UTC'
                        WHERE email = %s;
                        """,
                        (otp, expiration, payload.email.lower())
                    )
                    conn.commit()
                    try:
                        send_verification_email(payload.email.lower(), otp)
                    except Exception as mail_err:
                        print(f"⚠️  Email send failed: {mail_err}")
                    return AuthResponse(
                        success=True,
                        requires_verification=True,
                        message="A new verification code has been sent to your email."
                    )

            # Hash password
            password_hash = bcrypt.hashpw(
                payload.password.encode("utf-8"),
                bcrypt.gensalt(rounds=12)
            ).decode("utf-8")

            # Generate OTP
            otp = generate_otp()
            expiration = get_expiration()

            # Insert the new user
            cur.execute(
                """
                INSERT INTO users
                    (username, email, password_hash, verification_code, code_expiration,
                     is_verified, otp_attempts, last_otp_sent_at)
                VALUES (%s, %s, %s, %s, %s, FALSE, 0, NOW() AT TIME ZONE 'UTC')
                RETURNING id, username, email, created_at, is_verified;
                """,
                (payload.username.strip(), payload.email.lower(), password_hash, otp, expiration)
            )
            user = dict(cur.fetchone())
            conn.commit()

        # Send email (outside the DB cursor — don't hold the connection open)
        try:
            send_verification_email(payload.email.lower(), otp)
        except Exception as mail_err:
            print(f"⚠️  Email send failed: {mail_err}")
            # Registration still succeeds — user can use Resend

        return AuthResponse(
            success=True,
            requires_verification=True,
            message="Account created. Please check your email for the verification code."
        )

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        print(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during registration. Please try again."
        )
    finally:
        release_connection(conn)


# ─────────────────────────────────────────────
# POST /api/auth/login
# ─────────────────────────────────────────────

@router.post("/login", response_model=AuthResponse)
async def login(payload: LoginRequest):
    """
    Login with email and password.
    - Verifies password against bcrypt hash
    - Blocks login for unverified accounts
    - Returns JWT + user data on success
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, username, email, password_hash, created_at, is_verified, is_admin
                FROM users WHERE email = %s;
                """,
                (payload.email.lower(),)
            )
            user = cur.fetchone()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password."
            )

        user = dict(user)

        # Verify password — bcrypt.checkpw is timing-safe
        password_valid = bcrypt.checkpw(
            payload.password.encode("utf-8"),
            user["password_hash"].encode("utf-8")
        )

        if not password_valid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password."
            )

        # Block unverified accounts
        if not user.get("is_verified", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Please verify your email before logging in. Check your inbox for the verification code."
            )

        token = create_access_token(user["id"], user["email"], user.get("is_admin", False))
        return AuthResponse(
            success=True,
            token=token,
            user=UserResponse(
                id=user["id"],
                username=user["username"],
                email=user["email"],
                created_at=user["created_at"],
                is_verified=user["is_verified"],
                is_admin=user.get("is_admin", False)
            )
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during login. Please try again."
        )
    finally:
        release_connection(conn)


# ─────────────────────────────────────────────
# POST /api/auth/verify-otp
# ─────────────────────────────────────────────

@router.post("/verify-otp", response_model=AuthResponse)
async def verify_otp(payload: VerifyOTPRequest):
    """
    Verify the OTP code submitted by the user.
    - Checks attempt limit (max 5)
    - Checks code match
    - Checks code expiry
    - On success: marks account verified, clears OTP, issues JWT
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, username, email, created_at,
                       verification_code, code_expiration, is_verified, otp_attempts
                FROM users WHERE email = %s;
                """,
                (payload.email.lower(),)
            )
            user = cur.fetchone()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No account found with this email address."
            )

        user = dict(user)

        # Already verified?
        if user["is_verified"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This account is already verified. Please log in."
            )

        # Too many attempts?
        if user["otp_attempts"] >= MAX_OTP_ATTEMPTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many incorrect attempts. Please request a new verification code."
            )

        # Code expired?
        if is_expired(user["code_expiration"]):
            raise HTTPException(
                status_code=status.HTTP_410_GONE,
                detail="Verification code has expired. Please request a new one."
            )

        # Wrong code? Increment attempts
        if user["verification_code"] != payload.code:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE users SET otp_attempts = otp_attempts + 1 WHERE email = %s;",
                    (payload.email.lower(),)
                )
                conn.commit()
            remaining = MAX_OTP_ATTEMPTS - (user["otp_attempts"] + 1)
            if remaining <= 0:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many incorrect attempts. Please request a new verification code."
                )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Incorrect verification code. {remaining} attempt(s) remaining."
            )

        # ✅ Code is valid — verify account, clear OTP fields, issue JWT
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                UPDATE users
                SET is_verified       = TRUE,
                    verification_code = NULL,
                    code_expiration   = NULL,
                    otp_attempts      = 0
                WHERE email = %s
                RETURNING id, username, email, created_at, is_verified;
                """,
                (payload.email.lower(),)
            )
            verified_user = dict(cur.fetchone())
            # Fetch is_admin separately (newly verified users default to False)
            verified_user.setdefault("is_admin", False)
            conn.commit()

        token = create_access_token(verified_user["id"], verified_user["email"], verified_user.get("is_admin", False))
        return AuthResponse(
            success=True,
            token=token,
            user=UserResponse(**verified_user),
            requires_verification=False,
            message="Email verified successfully! Welcome to CareerPath AI."
        )

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        print(f"OTP verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during verification. Please try again."
        )
    finally:
        release_connection(conn)


# ─────────────────────────────────────────────
# POST /api/auth/resend-otp
# ─────────────────────────────────────────────

@router.post("/resend-otp", response_model=AuthResponse)
async def resend_otp(payload: ResendOTPRequest):
    """
    Resend a fresh OTP code to the user's email.
    - Rate limited: once every 60 seconds
    - Resets otp_attempts counter
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, is_verified, last_otp_sent_at FROM users WHERE email = %s;",
                (payload.email.lower(),)
            )
            user = cur.fetchone()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No account found with this email address."
            )

        user = dict(user)

        if user["is_verified"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This account is already verified."
            )

        # Rate limit check
        if not is_resend_allowed(user["last_otp_sent_at"]):
            remaining = seconds_until_resend_allowed(user["last_otp_sent_at"])
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Please wait {remaining} second(s) before requesting a new code."
            )

        # Generate new OTP
        otp = generate_otp()
        expiration = get_expiration()

        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE users
                SET verification_code = %s,
                    code_expiration   = %s,
                    otp_attempts      = 0,
                    last_otp_sent_at  = NOW() AT TIME ZONE 'UTC'
                WHERE email = %s;
                """,
                (otp, expiration, payload.email.lower())
            )
            conn.commit()

        # Send new email
        try:
            send_verification_email(payload.email.lower(), otp)
        except Exception as mail_err:
            print(f"⚠️  Email resend failed: {mail_err}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to send verification email. Please try again later."
            )

        return AuthResponse(
            success=True,
            requires_verification=True,
            message="A new verification code has been sent to your email."
        )

    except HTTPException:
        raise
    except Exception as e:
        conn.rollback()
        print(f"Resend OTP error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred. Please try again."
        )
    finally:
        release_connection(conn)


# ─────────────────────────────────────────────
# GET /api/auth/me
# ─────────────────────────────────────────────

@router.get("/me", response_model=UserResponse)
async def get_me(request: Request):
    """
    Return current user profile from a valid JWT token.
    Requires: Authorization: Bearer <token>
    """
    payload = get_current_user_payload(request)
    user_id = int(payload["sub"])

    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, username, email, created_at, is_verified, is_admin FROM users WHERE id = %s;",
                (user_id,)
            )
            user = cur.fetchone()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User account not found."
            )

        return UserResponse(**dict(user))

    except HTTPException:
        raise
    except Exception as e:
        print(f"Get user error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching user data."
        )
    finally:
        release_connection(conn)
