"""
Email Service for CareerPath AI
Sends transactional emails via Brevo (https://www.brevo.com).
Reads credentials from environment variables.
"""

import json
import os
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from dotenv import load_dotenv

load_dotenv()

BREVO_API_KEY = os.getenv("BREVO_API_KEY", "")
FROM_ADDRESS = os.getenv("BREVO_FROM", "CareerPath AI <info@careerpathai.tech>")
CURRENT_YEAR = datetime.now().year


def _send_email(to_email: str, subject: str, body_text: str, body_html: str | None = None) -> None:
  """
  Internal helper -- sends an email via Brevo.
  Raises an exception if sending fails (caller should handle).
  """
  if not BREVO_API_KEY:
    raise ValueError("BREVO_API_KEY is not set")

  payload = {
    "sender": {"email": FROM_ADDRESS},
    "to": [{"email": to_email}],
    "subject": subject,
    "textContent": body_text,
  }

  if body_html:
    payload["htmlContent"] = body_html

  request = Request(
    url="https://api.brevo.com/v3/smtp/email",
    data=json.dumps(payload).encode("utf-8"),
    headers={
      "accept": "application/json",
      "api-key": BREVO_API_KEY,
      "content-type": "application/json",
    },
    method="POST",
  )

  print(f"📧 Sending email via Brevo to {to_email} ...")

  try:
    with urlopen(request, timeout=20) as response:
      response_body = response.read().decode("utf-8")
      print(f"✅ Brevo response: {response.status} {response_body}")
  except HTTPError as exc:
    error_body = exc.read().decode("utf-8") if exc.fp else ""
    raise RuntimeError(f"Brevo API error {exc.code}: {error_body}") from exc
  except URLError as exc:
    raise RuntimeError(f"Brevo connection error: {exc}") from exc


def send_verification_email(to_email: str, otp_code: str) -> None:
    """
    Send a 6-digit OTP verification email to the given address.

    Args:
        to_email:  Recipient's email address
        otp_code:  The 6-digit OTP code to include in the email
    """
    subject = "Email Verification Code"

    body_text = (
        f"Your verification code is: {otp_code}\n\n"
        "This code will expire in 5 minutes.\n"
        "Do not share this code with anyone."
    )

    body_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Email Verification</title>
</head>
<body style="margin:0;padding:0;background-color:#f1f5f9;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f1f5f9;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="480" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
          <!-- Header -->
          <tr>
            <td style="background:linear-gradient(135deg,#2563eb,#3b82f6);padding:32px 40px;text-align:center;">
              <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;letter-spacing:-0.5px;">CareerPath AI</h1>
              <p style="margin:8px 0 0;color:rgba(255,255,255,0.85);font-size:14px;">Email Verification</p>
            </td>
          </tr>
          <!-- Body -->
          <tr>
            <td style="padding:40px;">
              <p style="margin:0 0 24px;color:#334155;font-size:15px;line-height:1.6;">
                Use the verification code below to confirm your email address. This code expires in <strong>5 minutes</strong>.
              </p>
              <!-- OTP Box -->
              <div style="background:#f8fafc;border:2px solid #e2e8f0;border-radius:12px;padding:28px;text-align:center;margin:0 0 28px;">
                <p style="margin:0 0 8px;color:#64748b;font-size:13px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Verification Code</p>
                <p style="margin:0;font-size:40px;font-weight:800;letter-spacing:12px;color:#1e293b;font-family:'Courier New',monospace;">{otp_code}</p>
              </div>
              <p style="margin:0 0 8px;color:#64748b;font-size:14px;line-height:1.6;">
                This code will expire in <strong>5 minutes</strong>.
              </p>
              <p style="margin:0;color:#94a3b8;font-size:13px;">
                Do not share this code with anyone. CareerPath AI will never ask for it.
              </p>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="border-top:1px solid #e2e8f0;padding:24px 40px;text-align:center;">
              <p style="margin:0;color:#94a3b8;font-size:12px;">
                If you did not create an account, you can safely ignore this email.
              </p>
              <p style="margin:8px 0 0;color:#94a3b8;font-size:12px;">
                &copy; {CURRENT_YEAR} CareerPath AI &bull; info@careerpathai.tech
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

    _send_email(to_email, subject, body_text, body_html)
    print(f"✅ Verification email sent to {to_email}")


def send_password_reset_email(to_email: str, otp_code: str) -> None:
    """
    Send a 6-digit OTP code for password reset.

    Args:
        to_email:  Recipient's email address
        otp_code:  The 6-digit OTP code to include in the email
    """
    subject = "Password Reset Code"

    body_text = (
        f"Your password reset code is: {otp_code}\n\n"
        "This code will expire in 5 minutes.\n"
        "If you did not request a password reset, you can safely ignore this email."
    )

    body_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Password Reset</title>
</head>
<body style="margin:0;padding:0;background-color:#f1f5f9;font-family:'Helvetica Neue',Helvetica,Arial,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#f1f5f9;padding:40px 0;">
    <tr>
      <td align="center">
        <table width="480" cellpadding="0" cellspacing="0" style="background:#ffffff;border-radius:16px;overflow:hidden;box-shadow:0 4px 24px rgba(0,0,0,0.08);">
          <!-- Header -->
          <tr>
            <td style="background:linear-gradient(135deg,#2563eb,#3b82f6);padding:32px 40px;text-align:center;">
              <h1 style="margin:0;color:#ffffff;font-size:22px;font-weight:700;letter-spacing:-0.5px;">CareerPath AI</h1>
              <p style="margin:8px 0 0;color:rgba(255,255,255,0.85);font-size:14px;">Password Reset</p>
            </td>
          </tr>
          <!-- Body -->
          <tr>
            <td style="padding:40px;">
              <p style="margin:0 0 24px;color:#334155;font-size:15px;line-height:1.6;">
                We received a request to reset your password. Use the code below to set a new password. This code expires in <strong>5 minutes</strong>.
              </p>
              <!-- OTP Box -->
              <div style="background:#f8fafc;border:2px solid #e2e8f0;border-radius:12px;padding:28px;text-align:center;margin:0 0 28px;">
                <p style="margin:0 0 8px;color:#64748b;font-size:13px;text-transform:uppercase;letter-spacing:1px;font-weight:600;">Reset Code</p>
                <p style="margin:0;font-size:40px;font-weight:800;letter-spacing:12px;color:#1e293b;font-family:'Courier New',monospace;">{otp_code}</p>
              </div>
              <p style="margin:0 0 8px;color:#64748b;font-size:14px;line-height:1.6;">
                This code will expire in <strong>5 minutes</strong>.
              </p>
              <p style="margin:0;color:#94a3b8;font-size:13px;">
                If you did not request this reset, you can safely ignore this email. Your password will not be changed.
              </p>
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="border-top:1px solid #e2e8f0;padding:24px 40px;text-align:center;">
              <p style="margin:0;color:#94a3b8;font-size:12px;">
                If you did not request a password reset, you can safely ignore this email.
              </p>
              <p style="margin:8px 0 0;color:#94a3b8;font-size:12px;">
                &copy; {CURRENT_YEAR} CareerPath AI &bull; info@careerpathai.tech
              </p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
"""

    _send_email(to_email, subject, body_text, body_html)
    print(f"✅ Password reset email sent to {to_email}")
