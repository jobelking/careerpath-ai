"""
Email Service for CareerPath AI
Sends transactional emails via SMTP (Titan Mail / careerpathai.tech).
Reads credentials from environment variables.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.titan.email")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
FROM_NAME = "CareerPath AI"
FROM_ADDRESS = SMTP_USER


def _send_email(to_email: str, subject: str, body_text: str, body_html: str | None = None) -> None:
    """
    Internal helper — tries STARTTLS first, falls back to SSL if port is 465.
    Raises an exception if sending fails (caller should handle).
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{FROM_NAME} <{FROM_ADDRESS}>"
    msg["To"] = to_email

    # Plain-text part (always included)
    msg.attach(MIMEText(body_text, "plain"))

    # HTML part (optional — used as the preferred rendering)
    if body_html:
        msg.attach(MIMEText(body_html, "html"))

    print(f"📧 Connecting to SMTP: {SMTP_HOST}:{SMTP_PORT} as {SMTP_USER}")

    if SMTP_PORT == 465:
        # SSL connection (no STARTTLS needed)
        import ssl
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=context, timeout=15) as server:
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(FROM_ADDRESS, to_email, msg.as_string())
    else:
        # STARTTLS connection (587 or custom)
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(FROM_ADDRESS, to_email, msg.as_string())


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
                &copy; 2025 CareerPath AI &bull; info@careerpathai.tech
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
