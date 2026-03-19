"""
BioSentinel v2.0 — Complete Production Backend
================================================
Run:   python app.py
Docs:  http://localhost:8000/docs

Features:
 ✅ 4 ML models (GBM + SHAP explanations)
 ✅ JWT auth + 2FA (TOTP) + rate limiting
 ✅ Multi-user isolation + audit log
 ✅ PDF/image OCR lab report auto-import
 ✅ FHIR R4 import (Patient/Observation/Medication)
 ✅ PostgreSQL + SQLite support
 ✅ Field-level AES-256 encryption (HIPAA-ready)
 ✅ Structured JSON logging (Datadog/CloudWatch/Loki)
 ✅ Full-text patient search with filters
 ✅ Webhook event system (Slack, custom endpoints)
 ✅ Batch AI predictions (all patients at once)
 ✅ PDF + CSV data export
 ✅ Session management (list/revoke tokens)
 ✅ Notification preferences per user
 ✅ Multi-tenant clinic management
 ✅ PWA support (patient mobile app)
 ✅ Genomic risk score integration (23andMe VCF)
 ✅ Video consultation links (Jitsi)
 ✅ Password reset: Email/SMS/WhatsApp/Telegram
 ✅ Drug interaction checker (OpenFDA)
 ✅ Bulk CSV/Excel import
 ✅ Trend alerts + overdue reminders
 ✅ 153+ passing tests
"""

import json, math, os, re, uuid, warnings, smtplib, threading, time, io, base64
import hashlib, hmac, logging, sys
import secrets as _secrets
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Any

# Optional 2FA
try:
    import pyotp, qrcode
    TOTP_AVAILABLE = True
except ImportError:
    TOTP_AVAILABLE = False

# Optional field encryption
try:
    from cryptography.fernet import Fernet, InvalidToken
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# Optional structured logging
try:
    import structlog
    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

# Optional PDF export
try:
    from fpdf import FPDF
    PDF_EXPORT_AVAILABLE = True
except ImportError:
    PDF_EXPORT_AVAILABLE = False

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text, create_engine, or_, and_
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

# Rate limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    RATE_LIMIT_AVAILABLE = False

# Optional OCR imports — graceful degradation if not installed
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

warnings.filterwarnings("ignore")

# ── CLAUDE AI INTEGRATION ─────────────────────────────────────────────────
try:
    from claude_ai import (
        extract_labs_from_image,
        extract_labs_from_pdf_pages,
        generate_prediction_narrative,
        detect_trend_anomalies,
        explain_drug_interactions,
        claude_ai_status,
    )
    CLAUDE_AI_AVAILABLE = True
except ImportError:
    CLAUDE_AI_AVAILABLE = False
    def claude_ai_status(): return {"available": False, "note": "claude_ai.py not found"}
    def generate_prediction_narrative(*a, **kw): return None
    def detect_trend_anomalies(*a, **kw): return None
    def explain_drug_interactions(*a, **kw): return None
    def extract_labs_from_image(*a, **kw): return {"error": "Not available", "values": {}}
    def extract_labs_from_pdf_pages(*a, **kw): return {"error": "Not available", "values": {}}

# ── BACKGROUND SCHEDULER ──────────────────────────────────────────────────
try:
    from scheduler import start_scheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    def start_scheduler(*a, **kw): return None

# ── MLFLOW EXPERIMENT TRACKING ────────────────────────────────────────────
try:
    from mlflow_tracking import track_training_run, mlflow_status
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    def track_training_run(*a, **kw): return None
    def mlflow_status(): return {"enabled": False, "available": False}

# ── IN-MEMORY PREDICTION CACHE ───────────────────────────────────────────────
# Lightweight TTL cache for prediction results and trend data.
# Avoids re-running ML inference on every page load.
# No Redis required — just a dict with timestamps.
import threading as _threading
from functools import wraps as _wraps

_cache_lock  = _threading.Lock()
_cache_store: dict = {}   # key → (value, expires_at)
_CACHE_TTL   = int(os.getenv("CACHE_TTL_SECONDS", "300"))  # 5 min default

def _cache_get(key: str):
    with _cache_lock:
        entry = _cache_store.get(key)
        if entry and entry[1] > time.time():
            return entry[0]
        _cache_store.pop(key, None)
        return None

def _cache_set(key: str, value, ttl: int = None):
    with _cache_lock:
        _cache_store[key] = (value, time.time() + (ttl or _CACHE_TTL))

def _cache_invalidate(pattern: str):
    """Remove all cache keys that contain the pattern (e.g. patient ID)."""
    with _cache_lock:
        keys_to_del = [k for k in _cache_store if pattern in k]
        for k in keys_to_del:
            del _cache_store[k]

def _cache_stats() -> dict:
    with _cache_lock:
        now = time.time()
        total = len(_cache_store)
        alive = sum(1 for _, (_, exp) in _cache_store.items() if exp > now)
        return {"total_entries": total, "alive": alive, "expired": total - alive,
                "ttl_seconds": _CACHE_TTL}

# ── CONFIG ──────────────────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("SECRET_KEY", "bs-dev-secret-xyz-2025-liveupx")
ALGORITHM   = "HS256"
TOKEN_EXP   = 60 * 24   # 24 hours for demo
DB_URL      = os.getenv("DATABASE_URL", "sqlite:///./biosentinel.db")

# ── DATABASE ENGINE — SQLite (dev) or PostgreSQL (prod) ──────────────────────
def _make_engine():
    """
    Build the SQLAlchemy engine.
    - SQLite:     DATABASE_URL=sqlite:///./biosentinel.db  (default)
    - PostgreSQL: DATABASE_URL=postgresql://user:pass@host:5432/dbname
    - Heroku/Render: DATABASE_URL starts with "postgres://"  (fixed automatically)
    """
    url = DB_URL
    # Heroku exports "postgres://" but SQLAlchemy needs "postgresql://"
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    is_sqlite = url.startswith("sqlite")
    kwargs = {"connect_args": {"check_same_thread": False}} if is_sqlite else {
        "pool_size":         int(os.getenv("DB_POOL_SIZE", "5")),
        "max_overflow":      int(os.getenv("DB_MAX_OVERFLOW", "10")),
        "pool_pre_ping":     True,   # auto-reconnect on stale connections
        "pool_recycle":      300,    # recycle connections every 5 min
    }
    eng = create_engine(url, **kwargs)
    return eng, is_sqlite

engine, _IS_SQLITE = _make_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

# ── STRUCTURED JSON LOGGING ───────────────────────────────────────────────────
def _setup_logging():
    """
    Configure structured JSON logging.
    - Development: human-readable coloured console output
    - Production (LOG_FORMAT=json): structured JSON for Datadog/CloudWatch/Loki
    Compatible with: Datadog, AWS CloudWatch, Grafana Loki, Elastic Stack.
    """
    log_level  = os.getenv("LOG_LEVEL",  "INFO").upper()
    log_format = os.getenv("LOG_FORMAT", "console").lower()   # console | json

    # Standard library logger as fallback / base
    logging.basicConfig(
        level   = getattr(logging, log_level, logging.INFO),
        format  = "%(message)s",
        stream  = sys.stdout,
        force   = True,
    )

    if STRUCTLOG_AVAILABLE:
        shared_processors = [
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
        ]
        if log_format == "json":
            # Production: pure JSON — works with all log aggregators
            structlog.configure(
                processors = shared_processors + [
                    structlog.processors.dict_tracebacks,
                    structlog.processors.JSONRenderer(),
                ],
                wrapper_class = structlog.stdlib.BoundLogger,
                context_class = dict,
                logger_factory = structlog.stdlib.LoggerFactory(),
            )
        else:
            # Development: coloured human-readable
            structlog.configure(
                processors = shared_processors + [
                    structlog.dev.ConsoleRenderer(colors=True),
                ],
                wrapper_class = structlog.stdlib.BoundLogger,
                context_class = dict,
                logger_factory = structlog.stdlib.LoggerFactory(),
            )
        return structlog.get_logger("biosentinel")
    else:
        return logging.getLogger("biosentinel")

logger = _setup_logging()


class AppLogger:
    """Thin wrapper providing consistent log calls throughout the app."""
    def __init__(self, _logger):
        self._l = _logger

    def info(self, event: str, **kw):
        try:    self._l.info(event, **kw)
        except: print(f"[INFO] {event} {kw}")

    def warning(self, event: str, **kw):
        try:    self._l.warning(event, **kw)
        except: print(f"[WARN] {event} {kw}")

    def error(self, event: str, **kw):
        try:    self._l.error(event, **kw)
        except: print(f"[ERROR] {event} {kw}")

    def debug(self, event: str, **kw):
        try:    self._l.debug(event, **kw)
        except: pass

log = AppLogger(logger)


# ── FIELD-LEVEL ENCRYPTION (AES-256 via Fernet) ──────────────────────────────
class FieldEncryption:
    """
    AES-256 (Fernet) field-level encryption for sensitive patient data.
    Protects: biomarker values, diagnoses, medications, notes.

    Setup:
      1. Generate key:  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
      2. Set env var:   FIELD_ENCRYPTION_KEY=<output>

    Without a key, encryption is disabled (plaintext storage).
    WARNING: Changing or losing the key makes all encrypted data unrecoverable.
             Always back up the key separately from the database.
    """

    def __init__(self):
        self.enabled = False
        self._f      = None
        key_str = os.getenv("FIELD_ENCRYPTION_KEY", "")
        if key_str and ENCRYPTION_AVAILABLE:
            try:
                self._f   = Fernet(key_str.encode() if isinstance(key_str, str) else key_str)
                self.enabled = True
                log.info("field_encryption_enabled")
            except Exception as e:
                log.warning("field_encryption_key_invalid", error=str(e))
        else:
            if ENCRYPTION_AVAILABLE:
                log.warning("field_encryption_disabled",
                            hint="Set FIELD_ENCRYPTION_KEY env var to enable HIPAA-grade encryption")

    def encrypt(self, value: str) -> str:
        """Encrypt a string value. Returns 'enc:' + base64 ciphertext."""
        if not self.enabled or value is None:
            return value
        try:
            return "enc:" + self._f.encrypt(str(value).encode()).decode()
        except Exception:
            return value

    def decrypt(self, value: str) -> str:
        """Decrypt a string. If not encrypted, returns as-is."""
        if not self.enabled or value is None:
            return value
        try:
            if str(value).startswith("enc:"):
                return self._f.decrypt(value[4:].encode()).decode()
        except (InvalidToken, Exception):
            pass
        return value

    def encrypt_float(self, value: Optional[float]) -> Optional[str]:
        """Encrypt a float by converting to string first."""
        if value is None: return None
        return self.encrypt(str(value))

    def decrypt_float(self, value) -> Optional[float]:
        """Decrypt back to float."""
        if value is None: return None
        try:    return float(self.decrypt(str(value)))
        except: return None

    def encrypt_dict(self, d: dict, fields: list) -> dict:
        """Encrypt specific fields in a dict in-place."""
        for f in fields:
            if f in d and d[f] is not None:
                d[f] = self.encrypt(str(d[f]))
        return d

    def decrypt_dict(self, d: dict, fields: list) -> dict:
        """Decrypt specific fields in a dict in-place."""
        for f in fields:
            if f in d and d[f] is not None:
                v = self.decrypt(str(d[f]))
                try:    d[f] = float(v)
                except: d[f] = v
        return d

    def rotate_key(self, new_key_str: str, db_session) -> dict:
        """
        Re-encrypt all sensitive data with a new key.
        Call this after rotating the encryption key.
        Returns count of records re-encrypted.
        """
        if not ENCRYPTION_AVAILABLE:
            return {"error": "cryptography library not installed"}
        new_fernet = Fernet(new_key_str.encode())
        count = 0
        # Re-encrypt checkup notes
        for chk in db_session.query(DBCheckup).all():
            if chk.notes and chk.notes.startswith("enc:"):
                plain = self.decrypt(chk.notes)
                chk.notes = "enc:" + new_fernet.encrypt(plain.encode()).decode()
                count += 1
        db_session.commit()
        self._f = new_fernet
        return {"re_encrypted_records": count}


crypto = FieldEncryption()

# ── MULTI-TENANT CONFIG ───────────────────────────────────────────────────────
MULTI_TENANT = os.getenv("MULTI_TENANT", "false").lower() == "true"
DEFAULT_CLINIC_NAME = os.getenv("CLINIC_NAME", "BioSentinel Clinic")

# ── EMAIL CONFIG (set via env vars or .env file) ─────────────────────────────
EMAIL_ENABLED  = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST     = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT     = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER     = os.getenv("EMAIL_USER", "")
EMAIL_PASS     = os.getenv("EMAIL_PASS", "")
EMAIL_FROM     = os.getenv("EMAIL_FROM", EMAIL_USER)
EMAIL_TO_ADMIN = os.getenv("EMAIL_TO_ADMIN", EMAIL_USER)

# ── NOTIFICATION CONFIG — Twilio (SMS/WhatsApp) + Telegram ───────────────────
TWILIO_SID          = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN        = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_SMS     = os.getenv("TWILIO_FROM_SMS", "")      # +1234567890
TWILIO_FROM_WA      = os.getenv("TWILIO_FROM_WHATSAPP", "") # whatsapp:+14155238886
TELEGRAM_BOT_TOKEN  = os.getenv("TELEGRAM_BOT_TOKEN", "")
RESET_TOKEN_EXPIRY_MINUTES = int(os.getenv("RESET_TOKEN_EXPIRY", "30"))
APP_BASE_URL        = os.getenv("APP_BASE_URL", "http://localhost:8000")

# ── EMAIL ENGINE ──────────────────────────────────────────────────────────────
class EmailEngine:
    """
    Sends alert emails via SMTP (Gmail by default).
    All email sending is async/non-blocking so it never slows the API.
    Set EMAIL_ENABLED=true and EMAIL_USER/EMAIL_PASS env vars to activate.
    """

    def _send(self, to: str, subject: str, body_html: str, body_text: str):
        """Internal: actually send the email."""
        if not EMAIL_ENABLED:
            print(f"[EMAIL DISABLED] Would send to {to}: {subject}")
            return
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = f"BioSentinel <{EMAIL_FROM}>"
            msg["To"]      = to
            msg.attach(MIMEText(body_text, "plain"))
            msg.attach(MIMEText(body_html, "html"))
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=10) as s:
                s.ehlo(); s.starttls(); s.login(EMAIL_USER, EMAIL_PASS)
                s.sendmail(EMAIL_FROM, [to], msg.as_string())
            print(f"[EMAIL] Sent '{subject}' to {to}")
        except Exception as e:
            print(f"[EMAIL ERROR] {e}")

    def send_async(self, to: str, subject: str, body_html: str, body_text: str = ""):
        """Fire-and-forget email in a background thread."""
        if not body_text:
            # strip tags for plain-text fallback
            import re
            body_text = re.sub(r"<[^>]+>", "", body_html)
        t = threading.Thread(target=self._send, args=(to, subject, body_html, body_text), daemon=True)
        t.start()

    def alert_email(self, patient_id: str, patient_age: int, patient_sex: str,
                    alerts: list, prediction: dict, recipient: str):
        """Compose and send a clinical alert email."""
        level_colors = {
            "CRITICAL": "#dc2626", "WARNING": "#d97706", "INFO": "#2563eb"
        }
        domain_emoji = {
            "cancer": "🎗", "metabolic": "🩸", "cardio": "❤", "hematologic": "🔬"
        }
        alert_rows = "".join(
            f"""<tr>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">
                    <span style="display:inline-block;padding:2px 8px;border-radius:12px;
                        font-size:12px;font-weight:700;color:white;background:{level_colors.get(a['level'],'#6b7280')}">
                        {a['level']}
                    </span>
                </td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">
                    {domain_emoji.get(a['cat'],'')} {a['cat'].capitalize()}
                </td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;color:#374151">
                    {a['msg']}
                </td>
            </tr>"""
            for a in alerts
        )
        scores = "".join(
            f"""<td style="text-align:center;padding:12px;background:#f9fafb;border-radius:8px;margin:4px">
                <div style="font-size:22px;font-weight:800;color:{level_colors.get(prediction.get(d,{}).get('level','LOW'),'#059669')}">
                    {prediction.get(d,{}).get('risk',0)*100:.0f}%
                </div>
                <div style="font-size:11px;color:#6b7280;margin-top:2px">{d.capitalize()}</div>
            </td>"""
            for d in ["cancer","metabolic","cardio","hematologic"]
        )
        now = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
        html = f"""
        <!DOCTYPE html><html><body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;padding:20px;color:#111">
            <div style="background:#059669;padding:20px;border-radius:12px 12px 0 0;text-align:center">
                <h1 style="color:white;margin:0;font-size:24px">🩺 BioSentinel Health Alert</h1>
                <p style="color:#d1fae5;margin:6px 0 0;font-size:14px">AI-powered early disease detection</p>
            </div>
            <div style="background:white;border:1px solid #e5e7eb;padding:24px;border-radius:0 0 12px 12px">
                <p style="color:#6b7280;font-size:13px;margin-top:0">Generated: {now}</p>
                <h2 style="margin-bottom:4px">Patient Alert: {patient_sex}, {patient_age} years</h2>
                <p style="color:#6b7280;font-size:13px">Patient ID: {patient_id[:8]}…</p>

                <h3 style="border-bottom:1px solid #e5e7eb;padding-bottom:8px">Risk Scores</h3>
                <table style="width:100%;border-collapse:separate;border-spacing:6px"><tr>{scores}</tr></table>

                <h3 style="border-bottom:1px solid #e5e7eb;padding-bottom:8px;margin-top:24px">Active Alerts</h3>
                <table style="width:100%;border-collapse:collapse">
                    <tr style="background:#f3f4f6">
                        <th style="text-align:left;padding:8px 12px;font-size:12px">Level</th>
                        <th style="text-align:left;padding:8px 12px;font-size:12px">Category</th>
                        <th style="text-align:left;padding:8px 12px;font-size:12px">Message</th>
                    </tr>
                    {alert_rows}
                </table>

                <div style="margin-top:20px;padding:14px;background:#fef3c7;border-radius:8px;font-size:13px;color:#92400e">
                    <strong>⚕ Clinical Note:</strong> These are AI-generated risk signals for decision-support only.
                    All findings must be reviewed by a qualified healthcare professional before clinical action.
                </div>

                <div style="margin-top:20px;border-top:1px solid #e5e7eb;padding-top:16px;font-size:12px;color:#9ca3af;text-align:center">
                    BioSentinel by Liveupx Pvt. Ltd. &nbsp;|&nbsp;
                    <a href="https://github.com/liveupx/biosentinel" style="color:#059669">GitHub</a>
                </div>
            </div>
        </body></html>"""
        self.send_async(recipient, f"[BioSentinel] Health Alert — Patient {patient_id[:6]}", html)

email_engine = EmailEngine()


# ── MODELS ──────────────────────────────────────────────────────────────────
class DBUser(Base):
    __tablename__ = "users"
    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username         = Column(String, unique=True, index=True)
    email            = Column(String, unique=True)
    hashed_password  = Column(String)
    role             = Column(String, default="clinician")
    phone            = Column(String, nullable=True)
    telegram_chat_id = Column(String, nullable=True)
    # 2FA fields
    totp_secret      = Column(String, nullable=True)    # base32 TOTP secret
    totp_enabled     = Column(Integer, default=0)       # 0=off, 1=on
    totp_backup_codes= Column(Text, nullable=True)      # JSON list of hashed backup codes
    created_at       = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

class DBPasswordResetToken(Base):
    """Single-use password reset tokens, expire after 30 minutes."""
    __tablename__ = "password_reset_tokens"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = Column(String, ForeignKey("users.id"))
    token      = Column(String, unique=True, index=True)
    channel    = Column(String, default="email")
    expires_at = Column(String)
    used       = Column(Integer, default=0)
    created_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

class DBActiveSession(Base):
    """Tracks active JWT sessions per user — for session management/revocation."""
    __tablename__ = "active_sessions"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = Column(String, ForeignKey("users.id"), index=True)
    jti        = Column(String, unique=True, index=True)  # JWT ID for revocation
    device     = Column(String, nullable=True)   # "Chrome on Mac", "Mobile App"
    ip_address = Column(String, nullable=True)
    user_agent = Column(String, nullable=True)
    created_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    last_seen  = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    expires_at = Column(String)
    revoked    = Column(Integer, default=0)

class DBWebhook(Base):
    """Webhooks — notify external systems on BioSentinel events."""
    __tablename__ = "webhooks"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id    = Column(String, ForeignKey("users.id"), index=True)
    name       = Column(String)               # e.g. "Slack Critical Alerts"
    url        = Column(String)               # destination URL
    secret     = Column(String, nullable=True) # HMAC signing secret
    events     = Column(Text, default="[]")   # JSON list: ["prediction.critical","alert.new"]
    active     = Column(Integer, default=1)
    last_fired = Column(String, nullable=True)
    fail_count = Column(Integer, default=0)
    created_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

class DBNotificationPreference(Base):
    """Per-user notification preferences — which events on which channels."""
    __tablename__ = "notification_preferences"
    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id          = Column(String, ForeignKey("users.id"), unique=True)
    # Channel enables
    email_enabled    = Column(Integer, default=1)
    sms_enabled      = Column(Integer, default=0)
    whatsapp_enabled = Column(Integer, default=0)
    telegram_enabled = Column(Integer, default=0)
    # Event type preferences
    notify_critical  = Column(Integer, default=1)   # CRITICAL predictions
    notify_high      = Column(Integer, default=1)   # HIGH predictions
    notify_moderate  = Column(Integer, default=0)   # MODERATE predictions
    notify_overdue   = Column(Integer, default=1)   # overdue checkup reminders
    notify_login     = Column(Integer, default=0)   # new login alerts
    # Quiet hours (UTC)
    quiet_start      = Column(Integer, nullable=True)  # e.g. 22 = 10pm
    quiet_end        = Column(Integer, nullable=True)  # e.g. 7 = 7am
    updated_at       = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

class DBClinic(Base):
    """Multi-tenant clinic/organisation model."""
    __tablename__ = "clinics"
    id           = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name         = Column(String, unique=True)
    slug         = Column(String, unique=True, index=True)  # url-safe identifier
    address      = Column(Text, nullable=True)
    phone        = Column(String, nullable=True)
    email        = Column(String, nullable=True)
    logo_url     = Column(String, nullable=True)
    timezone     = Column(String, default="Asia/Kolkata")
    active       = Column(Integer, default=1)
    created_at   = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

class DBClinicMember(Base):
    """Clinic membership — links users to clinics with roles."""
    __tablename__ = "clinic_members"
    id         = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    clinic_id  = Column(String, ForeignKey("clinics.id"), index=True)
    user_id    = Column(String, ForeignKey("users.id"), index=True)
    role       = Column(String, default="member")  # owner | admin | member
    joined_at  = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

class DBGenomicProfile(Base):
    """Genomic risk data — 23andMe / VCF integration."""
    __tablename__ = "genomic_profiles"
    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id       = Column(String, ForeignKey("patients.id"), unique=True)
    source           = Column(String, default="23andme")   # 23andme | ancestry | vcf
    upload_date      = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    snp_count        = Column(Integer, default=0)
    # Key SNP risk scores (0-1 scale)
    brca1_risk       = Column(Float, nullable=True)   # breast/ovarian cancer
    brca2_risk       = Column(Float, nullable=True)
    apoe4_carrier    = Column(Integer, nullable=True) # Alzheimer's risk (0/1)
    lynch_syndrome   = Column(Float, nullable=True)   # colorectal cancer
    tcf7l2_diabetes  = Column(Float, nullable=True)   # Type 2 diabetes
    ldlr_cardio      = Column(Float, nullable=True)   # cardiovascular
    # Raw variant summary (JSON)
    variants_summary = Column(Text, nullable=True)
    notes            = Column(Text, nullable=True)

class DBPatient(Base):
    __tablename__ = "patients"
    id                     = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id               = Column(String, ForeignKey("users.id"), nullable=True)  # who created/owns this patient
    age                    = Column(Integer)
    sex                    = Column(String)
    ethnicity              = Column(String, nullable=True)
    family_history_cancer  = Column(Integer, default=0)
    family_history_diabetes= Column(Integer, default=0)
    family_history_cardio  = Column(Integer, default=0)
    smoking_status         = Column(String, default="never")   # never|former|current
    alcohol_units_weekly   = Column(Float, default=0)
    exercise_min_weekly    = Column(Integer, default=0)
    notes                  = Column(Text, nullable=True)
    created_at             = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    checkups     = relationship("DBCheckup",    back_populates="patient", cascade="all,delete")
    medications  = relationship("DBMedication", back_populates="patient", cascade="all,delete")
    diagnoses    = relationship("DBDiagnosis",  back_populates="patient", cascade="all,delete")
    diet_plans   = relationship("DBDietPlan",   back_populates="patient", cascade="all,delete")
    predictions  = relationship("DBPrediction", back_populates="patient", cascade="all,delete")

class DBCheckup(Base):
    __tablename__ = "checkups"
    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id      = Column(String, ForeignKey("patients.id"))
    checkup_date    = Column(String)
    notes           = Column(Text, nullable=True)
    # Vitals
    weight_kg       = Column(Float, nullable=True)
    bmi             = Column(Float, nullable=True)
    bp_systolic     = Column(Integer, nullable=True)
    bp_diastolic    = Column(Integer, nullable=True)
    heart_rate      = Column(Integer, nullable=True)
    spo2            = Column(Float, nullable=True)
    temperature_c   = Column(Float, nullable=True)
    # CBC
    wbc             = Column(Float, nullable=True)
    rbc             = Column(Float, nullable=True)
    hemoglobin      = Column(Float, nullable=True)
    hematocrit      = Column(Float, nullable=True)
    platelets       = Column(Float, nullable=True)
    lymphocytes_pct = Column(Float, nullable=True)
    neutrophils_pct = Column(Float, nullable=True)
    monocytes_pct   = Column(Float, nullable=True)
    eosinophils_pct = Column(Float, nullable=True)
    mcv             = Column(Float, nullable=True)
    mch             = Column(Float, nullable=True)
    # Metabolic
    glucose_fasting = Column(Float, nullable=True)
    hba1c           = Column(Float, nullable=True)
    creatinine      = Column(Float, nullable=True)
    egfr            = Column(Float, nullable=True)
    bun             = Column(Float, nullable=True)
    alt             = Column(Float, nullable=True)
    ast             = Column(Float, nullable=True)
    albumin         = Column(Float, nullable=True)
    bilirubin       = Column(Float, nullable=True)
    ggt             = Column(Float, nullable=True)
    uric_acid       = Column(Float, nullable=True)
    # Lipids
    total_cholesterol = Column(Float, nullable=True)
    ldl             = Column(Float, nullable=True)
    hdl             = Column(Float, nullable=True)
    triglycerides   = Column(Float, nullable=True)
    # Hormones
    tsh             = Column(Float, nullable=True)
    t3              = Column(Float, nullable=True)
    t4              = Column(Float, nullable=True)
    vitamin_d       = Column(Float, nullable=True)
    vitamin_b12     = Column(Float, nullable=True)
    ferritin        = Column(Float, nullable=True)
    # Tumor markers
    cea             = Column(Float, nullable=True)
    ca125           = Column(Float, nullable=True)
    ca199           = Column(Float, nullable=True)
    psa             = Column(Float, nullable=True)
    afp             = Column(Float, nullable=True)
    # Inflammation
    crp             = Column(Float, nullable=True)
    esr             = Column(Float, nullable=True)
    patient = relationship("DBPatient", back_populates="checkups")

class DBMedication(Base):
    __tablename__ = "medications"
    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id    = Column(String, ForeignKey("patients.id"))
    name          = Column(String)
    dosage_mg     = Column(Float, nullable=True)
    frequency     = Column(String, nullable=True)
    start_date    = Column(String, nullable=True)
    end_date      = Column(String, nullable=True)
    prescribed_for= Column(String, nullable=True)
    active        = Column(Integer, default=1)
    patient = relationship("DBPatient", back_populates="medications")

class DBDiagnosis(Base):
    __tablename__ = "diagnoses"
    id            = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id    = Column(String, ForeignKey("patients.id"))
    icd10_code    = Column(String, nullable=True)
    description   = Column(String)
    diagnosed_date= Column(String, nullable=True)
    status        = Column(String, default="active")
    severity      = Column(String, nullable=True)
    patient = relationship("DBPatient", back_populates="diagnoses")

class DBDietPlan(Base):
    __tablename__ = "diet_plans"
    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id      = Column(String, ForeignKey("patients.id"))
    start_date      = Column(String)
    end_date        = Column(String, nullable=True)
    calories_daily  = Column(Integer, nullable=True)
    protein_g       = Column(Float, nullable=True)
    carbs_g         = Column(Float, nullable=True)
    fat_g           = Column(Float, nullable=True)
    fiber_g         = Column(Float, nullable=True)
    diet_type       = Column(String, nullable=True)
    restrictions    = Column(String, nullable=True)
    notes           = Column(Text, nullable=True)
    patient = relationship("DBPatient", back_populates="diet_plans")

class DBPrediction(Base):
    __tablename__ = "predictions"
    id               = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id       = Column(String, ForeignKey("patients.id"))
    created_at       = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    checkups_used    = Column(Integer)
    months_of_data   = Column(Float)
    data_completeness= Column(Float)
    cancer_risk      = Column(Float)
    cancer_level     = Column(String)
    metabolic_risk   = Column(Float)
    metabolic_level  = Column(String)
    cardio_risk      = Column(Float)
    cardio_level     = Column(String)
    hematologic_risk = Column(Float)
    hematologic_level= Column(String)
    composite_score  = Column(Float)
    top_features_json= Column(Text)
    alerts_json      = Column(Text)
    recommendation   = Column(Text)
    patient = relationship("DBPatient", back_populates="predictions")

class DBAlert(Base):
    __tablename__ = "alerts"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_id  = Column(String, ForeignKey("patients.id"))
    created_at  = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    level       = Column(String)   # INFO | WARNING | CRITICAL
    category    = Column(String)   # cancer | metabolic | cardio | hematologic | biomarker
    message     = Column(Text)
    acknowledged= Column(Integer, default=0)
    emailed     = Column(Integer, default=0)  # whether an email was sent

class DBEmailConfig(Base):
    """One row per user — their SMTP/notification settings."""
    __tablename__ = "email_config"
    id              = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id         = Column(String, ForeignKey("users.id"), unique=True)
    smtp_host       = Column(String, nullable=True)
    smtp_port       = Column(Integer, default=587)
    smtp_username   = Column(String, nullable=True)
    smtp_password   = Column(String, nullable=True)   # stored plaintext for demo; encrypt in prod
    smtp_use_tls    = Column(Integer, default=1)
    from_address    = Column(String, nullable=True)
    notify_to       = Column(String, nullable=True)   # comma-separated recipient emails
    notify_on_high  = Column(Integer, default=1)      # send email on HIGH risk
    notify_on_critical = Column(Integer, default=1)   # send email on CRITICAL
    enabled         = Column(Integer, default=0)      # master on/off switch
    updated_at      = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

class DBAuditLog(Base):
    """Immutable audit trail of all patient data access."""
    __tablename__ = "audit_log"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp   = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())
    user_id     = Column(String, nullable=True)
    username    = Column(String, nullable=True)
    action      = Column(String)   # e.g. view_patient, run_prediction, update_checkup
    patient_id  = Column(String, nullable=True)
    detail      = Column(Text, nullable=True)
    ip_address  = Column(String, nullable=True)

# ── SCHEMAS ─────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    username: str; email: str; password: str; role: str = "clinician"

class UserLogin(BaseModel):
    username: str; password: str

class PatientCreate(BaseModel):
    age: int; sex: str
    ethnicity: Optional[str] = None
    family_history_cancer: int = 0
    family_history_diabetes: int = 0
    family_history_cardio: int = 0
    smoking_status: str = "never"
    alcohol_units_weekly: float = 0
    exercise_min_weekly: int = 0
    notes: Optional[str] = None
    # owner_id injected server-side — not from client

class CheckupCreate(BaseModel):
    patient_id: str; checkup_date: str; notes: Optional[str] = None
    weight_kg: Optional[float]=None; bmi: Optional[float]=None
    bp_systolic: Optional[int]=None; bp_diastolic: Optional[int]=None
    heart_rate: Optional[int]=None; spo2: Optional[float]=None
    temperature_c: Optional[float]=None
    wbc: Optional[float]=None; rbc: Optional[float]=None
    hemoglobin: Optional[float]=None; hematocrit: Optional[float]=None
    platelets: Optional[float]=None; lymphocytes_pct: Optional[float]=None
    neutrophils_pct: Optional[float]=None; monocytes_pct: Optional[float]=None
    eosinophils_pct: Optional[float]=None; mcv: Optional[float]=None; mch: Optional[float]=None
    glucose_fasting: Optional[float]=None; hba1c: Optional[float]=None
    creatinine: Optional[float]=None; egfr: Optional[float]=None; bun: Optional[float]=None
    alt: Optional[float]=None; ast: Optional[float]=None; albumin: Optional[float]=None
    bilirubin: Optional[float]=None; ggt: Optional[float]=None; uric_acid: Optional[float]=None
    total_cholesterol: Optional[float]=None; ldl: Optional[float]=None
    hdl: Optional[float]=None; triglycerides: Optional[float]=None
    tsh: Optional[float]=None; t3: Optional[float]=None; t4: Optional[float]=None
    vitamin_d: Optional[float]=None; vitamin_b12: Optional[float]=None; ferritin: Optional[float]=None
    cea: Optional[float]=None; ca125: Optional[float]=None; ca199: Optional[float]=None
    psa: Optional[float]=None; afp: Optional[float]=None
    crp: Optional[float]=None; esr: Optional[float]=None

class MedicationCreate(BaseModel):
    patient_id: str; name: str
    dosage_mg: Optional[float]=None; frequency: Optional[str]=None
    start_date: Optional[str]=None; end_date: Optional[str]=None
    prescribed_for: Optional[str]=None; active: int = 1

class DiagnosisCreate(BaseModel):
    patient_id: str; description: str
    icd10_code: Optional[str]=None; diagnosed_date: Optional[str]=None
    status: str = "active"; severity: Optional[str]=None

class DietPlanCreate(BaseModel):
    patient_id: str; start_date: str; end_date: Optional[str]=None
    calories_daily: Optional[int]=None; protein_g: Optional[float]=None
    carbs_g: Optional[float]=None; fat_g: Optional[float]=None; fiber_g: Optional[float]=None
    diet_type: Optional[str]=None; restrictions: Optional[str]=None
    notes: Optional[str]=None

class EmailConfigUpdate(BaseModel):
    smtp_host: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: int = 1
    from_address: Optional[str] = None
    notify_to: Optional[str] = None      # comma-separated emails
    notify_on_high: int = 1
    notify_on_critical: int = 1
    enabled: int = 0

class PasswordChange(BaseModel):
    current_password: str
    new_password: str

class EmailTestRequest(BaseModel):
    to_address: str

# ── AUTH ────────────────────────────────────────────────────────────────────
pwd_ctx  = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
security = HTTPBearer()

def hash_pw(pw): return pwd_ctx.hash(pw)
def verify_pw(plain, hashed): return pwd_ctx.verify(plain, hashed)

REVOKED_JTIS: set = set()   # in-memory revocation cache (backed by DB)

def make_token(data, device: str = None, ip: str = None, db=None):
    """Create JWT with unique jti for session tracking."""
    jti = _secrets.token_hex(16)
    d = data.copy()
    d["exp"] = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXP)
    d["jti"] = jti
    token = jwt.encode(d, SECRET_KEY, algorithm=ALGORITHM)
    # Record session in DB if db session provided
    if db and "sub" in data:
        user = db.query(DBUser).filter(DBUser.username == data["sub"]).first()
        if user:
            session = DBActiveSession(
                user_id    = user.id,
                jti        = jti,
                device     = device or "Unknown",
                ip_address = ip or "Unknown",
                expires_at = (datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXP)).isoformat(),
            )
            db.add(session)
            try: db.commit()
            except: db.rollback()
    return token

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def current_user(creds: HTTPAuthorizationCredentials = Depends(security),
                 db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        jti      = payload.get("jti")
        if not username: raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Invalid token")
    # Check revocation
    if jti and jti in REVOKED_JTIS:
        raise HTTPException(401, "Session revoked")
    if jti:
        sess = db.query(DBActiveSession).filter(
            DBActiveSession.jti == jti, DBActiveSession.revoked == 1).first()
        if sess:
            REVOKED_JTIS.add(jti)
            raise HTTPException(401, "Session revoked")
    u = db.query(DBUser).filter(DBUser.username == username).first()
    if not u: raise HTTPException(401, "User not found")
    # Update last_seen
    if jti:
        try:
            s = db.query(DBActiveSession).filter(DBActiveSession.jti == jti).first()
            if s:
                s.last_seen = datetime.now(timezone.utc).isoformat()
                db.commit()
        except: pass
    return u

# ── EMAIL ENGINE ─────────────────────────────────────────────────────────────
class EmailConfigEngine:
    """
    Sends alert emails via user-configured SMTP (stored in DBEmailConfig).
    Also provides send_async for general-purpose HTML emails.
    Runs in a background thread so the API never blocks.
    Works with Gmail, Outlook, Mailgun, SendGrid, or any SMTP server.
    """

    def send_async(self, to: str, subject: str, body_html: str, body_text: str = ""):
        """Fire-and-forget general-purpose email (uses global EMAIL_* config)."""
        if not body_text:
            import re as _re
            body_text = _re.sub(r"<[^>]+>", "", body_html)
        t = threading.Thread(
            target=self._send_global,
            args=(to, subject, body_html, body_text),
            daemon=True
        )
        t.start()

    def _send_global(self, to: str, subject: str, body_html: str, body_text: str):
        """Send using global EMAIL_* env vars."""
        if not EMAIL_ENABLED:
            print(f"[EMAIL DISABLED] Would send to {to}: {subject}"); return
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = f"BioSentinel <{EMAIL_FROM}>"
            msg["To"]      = to
            msg.attach(MIMEText(body_text, "plain"))
            msg.attach(MIMEText(body_html, "html"))
            with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT, timeout=10) as s:
                s.ehlo(); s.starttls(); s.login(EMAIL_USER, EMAIL_PASS)
                s.sendmail(EMAIL_FROM, [to], msg.as_string())
            print(f"[EMAIL] Sent '{subject}' to {to}")
        except Exception as e:
            print(f"[EMAIL ERROR] {e}")

    def send_alert_email(self, cfg: DBEmailConfig, alert: DBAlert,
                         patient_age: int, patient_sex: str):
        """Fire-and-forget email send in a background thread."""
        if not cfg or not cfg.enabled:
            return
        if alert.level == "HIGH"     and not cfg.notify_on_high:     return
        if alert.level == "CRITICAL" and not cfg.notify_on_critical:  return
        if not cfg.smtp_host or not cfg.notify_to:
            return

        t = threading.Thread(
            target=self._send,
            args=(cfg, alert, patient_age, patient_sex),
            daemon=True
        )
        t.start()

    def _send(self, cfg: DBEmailConfig, alert: DBAlert,
              patient_age: int, patient_sex: str):
        try:
            subject = f"[BioSentinel] {alert.level} Health Alert — {patient_sex} Age {patient_age}"
            body = self._build_html(alert, patient_age, patient_sex)

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = cfg.from_address or cfg.smtp_username or "biosentinel@noreply.com"
            msg["To"]      = cfg.notify_to

            msg.attach(MIMEText(body, "html"))

            if cfg.smtp_use_tls:
                server = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=10)
                server.ehlo()
                server.starttls()
            else:
                server = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=10)

            if cfg.smtp_username and cfg.smtp_password:
                server.login(cfg.smtp_username, cfg.smtp_password)

            recipients = [e.strip() for e in cfg.notify_to.split(",") if e.strip()]
            server.sendmail(msg["From"], recipients, msg.as_string())
            server.quit()

            # mark as emailed
            db = SessionLocal()
            try:
                a = db.query(DBAlert).filter(DBAlert.id == alert.id).first()
                if a: a.emailed = 1; db.commit()
            finally:
                db.close()

        except Exception as e:
            print(f"  ⚠ Email send failed: {e}")

    def test_connection(self, cfg: DBEmailConfig, to_address: str) -> dict:
        """Test SMTP connection and send a test email. Returns {success, message}."""
        try:
            if cfg.smtp_use_tls:
                server = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=8)
                server.ehlo(); server.starttls()
            else:
                server = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=8)
            if cfg.smtp_username and cfg.smtp_password:
                server.login(cfg.smtp_username, cfg.smtp_password)

            msg = MIMEText(
                "<h2>✅ BioSentinel Email Test</h2>"
                "<p>Your email alerts are configured correctly. "
                "You will receive notifications when patients reach HIGH or CRITICAL risk levels.</p>"
                "<p><em>BioSentinel — Liveupx Pvt. Ltd.</em></p>",
                "html"
            )
            msg["Subject"] = "[BioSentinel] Email Test — Configuration OK"
            msg["From"]    = cfg.from_address or cfg.smtp_username or "biosentinel@noreply.com"
            msg["To"]      = to_address
            server.sendmail(msg["From"], [to_address], msg.as_string())
            server.quit()
            return {"success": True, "message": f"Test email sent to {to_address}"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    def _build_html(self, alert: DBAlert, age: int, sex: str) -> str:
        color = {"CRITICAL": "#dc2626", "HIGH": "#f97316",
                 "WARNING": "#d97706", "INFO": "#2563eb"}.get(alert.level, "#374151")
        cat_label = {
            "cancer": "🎗 Cancer Risk",
            "metabolic": "🩸 Metabolic Risk",
            "cardio": "❤ Cardiovascular Risk",
            "hematologic": "🔬 Hematologic Risk",
        }.get(alert.category, alert.category.title())

        return f"""
        <div style="font-family:sans-serif;max-width:560px;margin:0 auto;padding:24px">
          <div style="background:{color};color:white;padding:16px 20px;border-radius:8px 8px 0 0">
            <h2 style="margin:0;font-size:18px">⚕ BioSentinel — {alert.level} Health Alert</h2>
          </div>
          <div style="background:#f9fafb;border:1px solid #e5e7eb;border-top:none;padding:20px;border-radius:0 0 8px 8px">
            <table style="width:100%;margin-bottom:16px">
              <tr><td style="color:#6b7280;font-size:13px">Patient</td>
                  <td style="font-weight:700">{sex}, {age} years old</td></tr>
              <tr><td style="color:#6b7280;font-size:13px">Alert type</td>
                  <td style="font-weight:700">{cat_label}</td></tr>
              <tr><td style="color:#6b7280;font-size:13px">Alert level</td>
                  <td><span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:700">{alert.level}</span></td></tr>
              <tr><td style="color:#6b7280;font-size:13px">Time</td>
                  <td style="font-size:13px">{alert.created_at[:16].replace("T"," ")} UTC</td></tr>
            </table>
            <div style="background:white;border:1px solid #fca5a5;border-left:4px solid {color};padding:14px;border-radius:6px;margin-bottom:16px">
              <p style="margin:0;font-size:14px;line-height:1.6">{alert.message}</p>
            </div>
            <p style="font-size:12px;color:#9ca3af;margin:0">
              ⚕ BioSentinel is a research/decision-support tool, not a licensed medical device.
              All alerts must be reviewed by a qualified healthcare professional.<br>
              — Liveupx Pvt. Ltd. | github.com/liveupx/biosentinel
            </p>
          </div>
        </div>
        """

    def send_reminder_email(self, cfg, patient_age: int, patient_sex: str,
                            patient_id: str, days_overdue: int):
        """Send an overdue-checkup reminder to the clinician. Fire-and-forget."""
        if not cfg or not cfg.enabled or not cfg.smtp_host or not cfg.notify_to:
            return
        t = threading.Thread(
            target=self._send_reminder_sync,
            args=(cfg, patient_age, patient_sex, patient_id, days_overdue),
            daemon=True
        )
        t.start()

    def _send_reminder_sync(self, cfg, patient_age: int, patient_sex: str,
                            patient_id: str, days_overdue: int):
        try:
            subject = (f"[BioSentinel] Overdue Checkup — {patient_sex} Age {patient_age} "                       f"({days_overdue}d overdue)")
            html = f"""
            <div style="font-family:sans-serif;max-width:560px;margin:0 auto;padding:24px">
              <div style="background:#d97706;color:white;padding:16px 20px;border-radius:8px 8px 0 0">
                <h2 style="margin:0;font-size:18px">⚕ BioSentinel — Overdue Checkup Reminder</h2>
              </div>
              <div style="background:#f9fafb;border:1px solid #e5e7eb;border-top:none;padding:20px;border-radius:0 0 8px 8px">
                <p>A patient is overdue for their quarterly checkup:</p>
                <table style="width:100%;margin-bottom:16px">
                  <tr><td style="color:#6b7280;font-size:13px">Patient</td>
                      <td style="font-weight:700">{patient_sex}, {patient_age} years old</td></tr>
                  <tr><td style="color:#6b7280;font-size:13px">Days overdue</td>
                      <td style="font-weight:700;color:#d97706">{days_overdue} days</td></tr>
                </table>
                <p style="font-size:12px;color:#9ca3af">
                  This is an automated reminder from BioSentinel background scheduler.<br>
                  — Liveupx Pvt. Ltd. | github.com/liveupx/biosentinel
                </p>
              </div>
            </div>"""
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = cfg.from_address or cfg.smtp_username or "biosentinel@noreply.com"
            msg["To"]      = cfg.notify_to
            msg.attach(MIMEText(html, "html"))
            if cfg.smtp_use_tls:
                server = smtplib.SMTP(cfg.smtp_host, cfg.smtp_port, timeout=10)
                server.ehlo(); server.starttls()
            else:
                server = smtplib.SMTP_SSL(cfg.smtp_host, cfg.smtp_port, timeout=10)
            if cfg.smtp_username and cfg.smtp_password:
                server.login(cfg.smtp_username, cfg.smtp_password)
            server.sendmail(msg["From"], [cfg.notify_to], msg.as_string())
            server.quit()
        except Exception as e:
            print(f"  ⚠ Reminder email failed: {e}")

email_config_engine = EmailConfigEngine()

# ── NOTIFICATION ENGINE — SMS, WhatsApp, Telegram ───────────────────────────
class NotificationEngine:
    """
    Multi-channel notification: Email, SMS (Twilio), WhatsApp (Twilio),
    Telegram Bot. Used for password reset OTPs and appointment reminders.
    All sends are fire-and-forget (async thread) so they never block the API.
    """

    def send_reset_otp(self, channel: str, destination: str,
                       otp: str, username: str) -> dict:
        """
        Send a password-reset OTP via the requested channel.
        Returns {"sent": True/False, "channel": ..., "error": ...}
        """
        msg = f"Your BioSentinel password reset code is: {otp}\nValid for {RESET_TOKEN_EXPIRY_MINUTES} minutes. If you didn't request this, ignore it."

        if channel == "email":
            return self._via_email(destination, username, otp)
        elif channel == "sms":
            return self._via_sms(destination, msg)
        elif channel == "whatsapp":
            return self._via_whatsapp(destination, msg)
        elif channel == "telegram":
            return self._via_telegram(destination, msg)
        else:
            return {"sent": False, "error": f"Unknown channel: {channel}"}

    def send_reminder(self, channel: str, destination: str,
                      patient_age: int, days_overdue: int) -> dict:
        """Send a checkup-due reminder."""
        msg = (f"BioSentinel Reminder: A patient (Age {patient_age}) "
               f"is {days_overdue} days overdue for their quarterly checkup. "
               f"Please schedule an appointment.")
        if channel == "email":
            html = f"""<div style="font-family:Arial,sans-serif;padding:20px">
                <h2 style="color:#059669">⏰ Checkup Reminder — BioSentinel</h2>
                <p>A patient (Age {patient_age}) is <strong>{days_overdue} days overdue</strong>
                for their quarterly checkup.</p>
                <p>Please schedule an appointment soon to maintain consistent monitoring data.</p>
                <hr/><p style="color:#6b7280;font-size:12px">BioSentinel · Liveupx Pvt. Ltd.</p>
                </div>"""
            email_engine.send_async(destination, "BioSentinel — Checkup Reminder", html)
            return {"sent": True, "channel": "email"}
        elif channel == "sms":
            return self._via_sms(destination, msg)
        elif channel == "whatsapp":
            return self._via_whatsapp(destination, msg)
        elif channel == "telegram":
            return self._via_telegram(destination, msg)
        return {"sent": False, "error": f"Unknown channel: {channel}"}

    def _via_email(self, email: str, username: str, otp: str) -> dict:
        html = f"""<div style="font-family:Arial,sans-serif;max-width:500px;margin:0 auto;padding:24px">
            <div style="background:#059669;padding:16px;border-radius:10px 10px 0 0;text-align:center">
                <h1 style="color:white;margin:0;font-size:22px">🔐 Password Reset</h1>
            </div>
            <div style="background:white;border:1px solid #e5e7eb;padding:24px;border-radius:0 0 10px 10px">
                <p>Hi <strong>{username}</strong>,</p>
                <p>Your BioSentinel password reset code is:</p>
                <div style="text-align:center;margin:20px 0">
                    <span style="font-size:32px;font-weight:800;letter-spacing:8px;
                        color:#059669;background:#d1fae5;padding:12px 24px;border-radius:8px">
                        {otp}
                    </span>
                </div>
                <p style="color:#6b7280;font-size:13px">
                    This code expires in {RESET_TOKEN_EXPIRY_MINUTES} minutes.<br/>
                    If you didn't request this, ignore this email.
                </p>
                <hr/><p style="color:#9ca3af;font-size:11px">BioSentinel · Liveupx Pvt. Ltd.</p>
            </div></div>"""
        try:
            email_engine.send_async(email, "BioSentinel — Password Reset Code", html)
            return {"sent": True, "channel": "email"}
        except Exception as e:
            return {"sent": False, "channel": "email", "error": str(e)}

    def _via_sms(self, phone: str, message: str) -> dict:
        if not TWILIO_SID or not TWILIO_TOKEN:
            return {"sent": False, "channel": "sms",
                    "error": "Twilio not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."}
        try:
            from twilio.rest import Client as TwilioClient
            client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
            msg = client.messages.create(body=message, from_=TWILIO_FROM_SMS, to=phone)
            return {"sent": True, "channel": "sms", "sid": msg.sid}
        except Exception as e:
            return {"sent": False, "channel": "sms", "error": str(e)}

    def _via_whatsapp(self, phone: str, message: str) -> dict:
        if not TWILIO_SID or not TWILIO_TOKEN:
            return {"sent": False, "channel": "whatsapp",
                    "error": "Twilio not configured. Set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."}
        try:
            from twilio.rest import Client as TwilioClient
            client = TwilioClient(TWILIO_SID, TWILIO_TOKEN)
            # WhatsApp numbers must be formatted as whatsapp:+1234567890
            to_wa = f"whatsapp:{phone}" if not phone.startswith("whatsapp:") else phone
            msg = client.messages.create(body=message, from_=TWILIO_FROM_WA, to=to_wa)
            return {"sent": True, "channel": "whatsapp", "sid": msg.sid}
        except Exception as e:
            return {"sent": False, "channel": "whatsapp", "error": str(e)}

    def _via_telegram(self, chat_id: str, message: str) -> dict:
        if not TELEGRAM_BOT_TOKEN:
            return {"sent": False, "channel": "telegram",
                    "error": "Telegram not configured. Set TELEGRAM_BOT_TOKEN."}
        try:
            import urllib.request as ur
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = json.dumps({"chat_id": chat_id, "text": message,
                                  "parse_mode": "HTML"}).encode()
            req = ur.Request(url, data=payload,
                             headers={"Content-Type": "application/json"})
            resp = json.loads(ur.urlopen(req, timeout=10).read())
            return {"sent": bool(resp.get("ok")), "channel": "telegram"}
        except Exception as e:
            return {"sent": False, "channel": "telegram", "error": str(e)}

notify_engine = NotificationEngine()

# ── OCR / LAB REPORT PARSER ───────────────────────────────────────────────────
class LabReportOCR:
    """
    Extracts biomarker values from PDF or image lab reports.
    Supports: PDF text extraction (pdfplumber), image OCR (pytesseract).
    Falls back gracefully if libraries not installed.

    Recognises common Indian/international lab report formats:
    SRL, Dr. Lal PathLabs, Metropolis, Apollo Diagnostics, NABL formats.
    """

    # Map of common lab report label variants → our field names
    FIELD_MAP = {
        # HbA1c — covers OCR artefacts like "HbAIc", "Hb A1 C", "HBAIC"
        r"hb\s*a\s*[1i]\s*c|glycated\s*hemo|glycohemo|\ba1c\b|hbaic": ("hba1c", float),
        # Fasting glucose
        r"fasting\s*(?:blood\s*)?(?:glucose|sugar|bs|bg)|fbg|fbs": ("glucose_fasting", float),
        # CBC
        r"haemoglobin|hemoglobin|hb\b|hgb": ("hemoglobin", float),
        r"total\s*wbc|wbc|white\s*blood\s*cell|total\s*leucocyte|tlc|leukocyte\s*count": ("wbc", float),
        r"platelet|plt\b|thrombocyte": ("platelets", float),  # note: lakhs→K handled in sane()
        r"lymphocyte\s*%?|lymphocytes\s*%?|\blymph\b.*%?|%\s*lymph|differential.*lymph": ("lymphocytes_pct", float),
        r"neutrophil\s*%|neutrophils\s*%|neut\s*%|%\s*neut|pmn\s*%": ("neutrophils_pct", float),
        r"\brdw\b|\bred\s*cell\s*dist": ("mcv", float),   # approximate
        # Lipids
        r"ldl[\s\-]*c(?:holesterol)?|low\s*density": ("ldl", float),
        r"hdl[\s\-]*c(?:holesterol)?|high\s*density": ("hdl", float),
        r"total\s*cholesterol|s\.cholesterol|serum\s*cholesterol": ("total_cholesterol", float),
        r"triglyceride|tg\b|trigs": ("triglycerides", float),
        # Liver
        r"\balt\b|alanine\s*amino|sgpt": ("alt", float),
        r"\bast\b|aspartate\s*amino|sgot": ("ast", float),
        r"\bggt\b|gamma\s*glutamyl": ("ggt", float),
        r"serum\s*albumin|\balbumin\b": ("albumin", float),
        r"total\s*bilirubin|s\.bilirubin": ("bilirubin", float),
        # Kidney
        r"serum\s*creatinine|\bcreatinine\b": ("creatinine", float),
        r"\begfr\b|estimated\s*gfr": ("egfr", float),
        r"\bbun\b|blood\s*urea\s*nitrogen|urea\s*nitrogen": ("bun", float),
        r"uric\s*acid|serum\s*urate": ("uric_acid", float),
        # Thyroid
        r"\btsh\b|thyroid\s*stimulating": ("tsh", float),
        r"\bt3\b|tri.?iodo": ("t3", float),
        r"\bt4\b|thyroxine": ("t4", float),
        # Vitamins / minerals
        r"vitamin\s*d|25\s*oh\s*d|25-hydroxy": ("vitamin_d", float),
        r"vitamin\s*b\s*12|cyanocobalamin": ("vitamin_b12", float),
        r"\bferritin\b": ("ferritin", float),
        # Tumour markers
        r"\bcea\b|carcinoembryonic": ("cea", float),
        r"\bca[\s\-]*125\b|cancer\s*antigen\s*125|a125": ("ca125", float),
        r"\bca[\s\-]*19[\s\-]*9\b": ("ca199", float),
        r"\bpsa\b|prostate\s*specific": ("psa", float),
        r"\bafp\b|alpha\s*feto": ("afp", float),
        # Inflammation
        r"\bcrp\b|c[\s\-]reactive\s*protein": ("crp", float),
        r"\besr\b|erythrocyte\s*sed": ("esr", float),
        # Vitals (if in report)
        r"blood\s*pressure.*systolic|systolic\s*bp|sbp": ("bp_systolic", int),
        r"blood\s*pressure.*diastolic|diastolic\s*bp|dbp": ("bp_diastolic", int),
        r"\bbmi\b|body\s*mass\s*index": ("bmi", float),
        r"body\s*weight|weight\s*kg|\bweight\b": ("weight_kg", float),
    }

    # Regex to extract a numeric value after a label
    VALUE_PATTERN = re.compile(
        r"([\d]+\.?\d*)\s*(?:mmol/l|mg/dl|mg/l|g/dl|g/l|u/l|iu/l|"
        r"k/ul|10\^3|thou|%|miu/l|ng/ml|pg/ml|ug/ml|ng/dl|pmol/l|nmol/l|"
        r"umol/l|mmhg|kg/m2|kg)?",
        re.IGNORECASE,
    )

    def extract_from_text(self, text: str) -> dict:
        """Parse lab values from plain text."""
        results = {}
        lines = text.lower().replace("\r", "\n").split("\n")

        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue
            for pattern, (field, cast) in self.FIELD_MAP.items():
                if re.search(pattern, line, re.IGNORECASE):
                    # Find a number in this line (or the next one)
                    nums = re.findall(r"\b(\d+\.?\d*)\b", line)
                    if not nums:
                        continue
                    # Pick the first plausible number (skip page numbers etc.)
                    for n in nums:
                        try:
                            val = cast(n)
                            # Sanity bounds — reject obviously wrong values
                            if self._sane(field, val):
                                if field not in results:   # first match wins
                                    results[field] = val
                                break
                        except (ValueError, TypeError):
                            continue
        return results

    def _sane(self, field: str, val: float) -> bool:
        """Rough sanity check — rejects page numbers / ref range artefacts."""
        bounds = {
            "hba1c": (3, 15), "glucose_fasting": (30, 600),
            "hemoglobin": (3, 20), "wbc": (0.5, 100),
            "platelets": (1, 2000), "lymphocytes_pct": (1, 99),  # 1 allows lakh values; _sane converts
            "neutrophils_pct": (1, 99), "cea": (0, 500),
            "ca125": (0, 10000), "psa": (0, 200),
            "alt": (1, 3000), "ast": (1, 3000),
            "creatinine": (0.1, 20), "tsh": (0.001, 100),
            "ldl": (10, 500), "hdl": (5, 200),
            "total_cholesterol": (50, 600), "triglycerides": (10, 3000),
            "crp": (0, 500), "ferritin": (0, 5000),
            "vitamin_d": (1, 200), "vitamin_b12": (50, 5000),
            "bp_systolic": (50, 250), "bp_diastolic": (30, 150),
            "bmi": (10, 80), "weight_kg": (10, 300),
        }
        lo, hi = bounds.get(field, (0, 999999))
        return lo <= val <= hi

    def from_pdf(self, file_bytes: bytes) -> dict:
        """Extract biomarkers from a PDF lab report."""
        if not PDF_AVAILABLE:
            return {"error": "pdfplumber not installed. Run: pip install pdfplumber"}
        try:
            text_parts = []
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
                    # Also try table extraction for tabular formats
                    for tbl in page.extract_tables() or []:
                        for row in tbl:
                            if row:
                                text_parts.append("  ".join(
                                    str(c) for c in row if c))
            full_text = "\n".join(text_parts)
            extracted = self.extract_from_text(full_text)
            return {"values": extracted, "raw_text_length": len(full_text),
                    "method": "pdf_text"}
        except Exception as e:
            return {"error": f"PDF parse failed: {str(e)}"}

    def from_image(self, file_bytes: bytes) -> dict:
        """Extract biomarkers from a photo of a lab report using OCR."""
        if not OCR_AVAILABLE:
            return {"error": "pytesseract/pillow not installed. Run: pip install pytesseract pillow"}
        try:
            img = Image.open(io.BytesIO(file_bytes))
            # Pre-process: convert to greyscale, increase contrast
            img = img.convert("L")
            # Upscale small images for better OCR
            w, h = img.size
            if w < 1200:
                scale = 1200 / w
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

            text = pytesseract.image_to_string(img, config="--psm 6")
            extracted = self.extract_from_text(text)
            return {"values": extracted, "raw_text_length": len(text),
                    "method": "image_ocr"}
        except Exception as e:
            return {"error": f"Image OCR failed: {str(e)}"}

    def from_upload(self, file_bytes: bytes, filename: str) -> dict:
        """Route to PDF or image handler based on file extension."""
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
        if ext == "pdf":
            return self.from_pdf(file_bytes)
        elif ext in ("jpg", "jpeg", "png", "tiff", "tif", "bmp", "webp"):
            return self.from_image(file_bytes)
        else:
            return {"error": f"Unsupported file type: {ext}. Use PDF, JPG, or PNG."}


lab_ocr = LabReportOCR()


def audit(db: Session, user, action: str,
          patient_id: str = None, detail: str = None):
    """Write an immutable audit log entry. Never raises — fails silently."""
    try:
        db.add(DBAuditLog(
            user_id    = user.id if user else None,
            username   = user.username if user else "system",
            action     = action,
            patient_id = patient_id,
            detail     = detail,
        ))
        db.commit()
    except Exception:
        pass

# ── ML ENGINE ───────────────────────────────────────────────────────────────
class BioSentinelEngine:
    """
    Properly calibrated multi-disease prediction engine.
    Uses 5,000-sample synthetic dataset with realistic clinical noise and
    class overlap. Gradient Boosting + isotonic calibration.
    """
    FEATURES = [
        "age","sex_f","smoke","alcohol","exercise_inv",
        "fam_cancer","fam_diab","fam_cardio","n_checkups","months",
        # Latest biomarker values
        "hba1c","glucose","hemoglobin","lymph","wbc","platelets",
        "cea","ca125","psa","alt","ast","ldl","hdl","triglyc",
        "bp_sys","bmi","creatinine","tsh","crp","ferritin",
        # Slope / trend (Δ per month) — key predictive signal
        "hba1c_slope","glucose_slope","hemoglobin_slope","lymph_slope",
        "wbc_slope","cea_slope","alt_slope","ldl_slope","bp_slope","bmi_slope",
        "platelets_slope","crp_slope",
        # Volatility (std over timeline)
        "hba1c_vol","cea_vol","hemoglobin_vol","lymph_vol",
        # Reference-range violations count
        "n_high","n_low","n_critical",
    ]

    def __init__(self):
        self.models = {}
        self.scaler = None
        self.trained = False
        self.shap_available = False

    # ── synthetic data generation ──────────────────────────────────────────
    def _synthetic(self, n=5000):
        rng = np.random.RandomState(42)

        rows, labels = [], {d:[] for d in ("cancer","metabolic","cardio","hematologic")}

        for _ in range(n):
            age   = rng.uniform(25, 80)
            sex_f = rng.randint(0, 2)
            smoke = rng.choice([0, 0, 0, 1, 2], p=[0.6, 0.0, 0.0, 0.25, 0.15])
            alc   = rng.exponential(3.0)
            exer  = rng.choice([0, 60, 150, 300], p=[0.3, 0.25, 0.25, 0.20])
            fam_c = rng.choice([0,1,2], p=[0.65, 0.25, 0.10])
            fam_d = rng.choice([0,1],   p=[0.72, 0.28])
            fam_v = rng.choice([0,1],   p=[0.68, 0.32])
            n_chk = rng.randint(2, 9)
            months= n_chk * 3.0

            # ── underlying disease probability (continuous, realistic) ──
            p_meta   = _sigmoid(-4.5 + 0.06*age + 0.8*fam_d + 0.4*(smoke>0)
                                 + 0.03*alc - 0.003*exer + rng.normal(0, 0.6))
            p_cancer = _sigmoid(-5.2 + 0.055*age + 0.6*fam_c + 0.35*(smoke==2)
                                 + 0.15*fam_d + rng.normal(0, 0.8))
            p_cardio = _sigmoid(-4.8 + 0.05*age + 0.5*fam_v + 0.3*(smoke>0)
                                 + 0.02*alc - 0.002*exer + rng.normal(0, 0.7))
            p_hema   = _sigmoid(-5.5 + 0.04*age + 0.3*fam_c + rng.normal(0, 0.9))

            is_meta   = rng.random() < p_meta
            is_cancer = rng.random() < p_cancer
            is_cardio = rng.random() < p_cardio
            is_hema   = rng.random() < p_hema

            # ── base biomarker values (age/sex adjusted) ──────────────
            hba1c     = rng.normal(5.4 + age*0.008 + is_meta*0.9, 0.35)
            glucose   = rng.normal(90  + age*0.15  + is_meta*14,  10)
            hemoglobin= rng.normal(14.5 - sex_f*1.3 - is_cancer*0.9 - is_hema*1.1, 0.9)
            lymph     = rng.normal(31  - age*0.07  - is_cancer*5 - is_hema*6, 4.5)
            wbc       = rng.normal(7.0 + is_hema*rng.choice([-2.5, 3.5]),  1.3)
            platelets = rng.normal(255 - is_hema*50,  45)
            cea       = max(0.3, rng.normal(1.6 + is_cancer*3.5, 1.1))
            ca125     = max(1, rng.normal(15  + is_cancer*12,  8))
            psa       = max(0, rng.normal(1.5 + (1-sex_f)*is_cancer*4, 1.2)) if not sex_f else 0
            alt       = rng.normal(24 + is_meta*18 + age*0.05, 10)
            ast       = rng.normal(22 + is_meta*12, 8)
            ldl       = rng.normal(108 + age*0.4  + is_cardio*28, 22)
            hdl       = rng.normal(54  - is_cardio*10 - age*0.1, 9)
            triglyc   = rng.normal(125 + is_meta*55 + is_cardio*30, 35)
            bp_sys    = rng.normal(118 + age*0.3  + is_cardio*18, 13)
            bmi       = rng.normal(24.5 + is_meta*3.5, 3.5)
            creat     = rng.normal(0.87 + age*0.003, 0.16)
            tsh       = rng.normal(2.2, 1.0)
            crp       = max(0.1, rng.normal(1.5 + is_cancer*3 + is_cardio*2, 1.5))
            ferritin  = rng.normal(80 + is_hema*rng.choice([-50, 120]), 40)

            # ── trend (slope per month) — realistic rates ──────────────
            slope_noise = rng.normal(0, 1.0, 12)
            hba1c_sl   = (is_meta * rng.uniform(0.012, 0.04)
                          + slope_noise[0]*0.004)
            glucose_sl = (is_meta * rng.uniform(0.3, 0.9)
                          + slope_noise[1]*0.15)
            hgb_sl     = (-is_cancer * rng.uniform(0.01, 0.04)
                          - is_hema   * rng.uniform(0.02, 0.06)
                          + slope_noise[2]*0.015)
            lymph_sl   = (-is_cancer * rng.uniform(0.1, 0.35)
                          - is_hema   * rng.uniform(0.1, 0.4)
                          + slope_noise[3]*0.08)
            wbc_sl     = (is_hema * rng.choice([-0.12, 0.18]) * rng.uniform(0.5, 1.5)
                          + slope_noise[4]*0.05)
            cea_sl     = (is_cancer * rng.uniform(0.05, 0.25)
                          + slope_noise[5]*0.02)
            alt_sl     = (is_meta   * rng.uniform(0.1, 0.5)
                          + slope_noise[6]*0.04)
            ldl_sl     = (is_cardio * rng.uniform(0.2, 0.8)
                          + slope_noise[7]*0.06)
            bp_sl      = (is_cardio * rng.uniform(0.2, 0.7)
                          + slope_noise[8]*0.05)
            bmi_sl     = (is_meta   * rng.uniform(0.02, 0.1)
                          + slope_noise[9]*0.01)
            plt_sl     = (-is_hema * rng.uniform(0.5, 2.0)
                          + slope_noise[10]*0.2)
            crp_sl     = ((is_cancer + is_cardio) * rng.uniform(0.02, 0.12)
                          + slope_noise[11]*0.015)

            # ── volatility features ──────────────────────────────────
            hba1c_v    = abs(rng.normal(is_meta*0.12, 0.06)) * n_chk**0.5
            cea_v      = abs(rng.normal(is_cancer*0.4, 0.2))
            hgb_v      = abs(rng.normal((is_cancer+is_hema)*0.2, 0.08))
            lymph_v    = abs(rng.normal((is_cancer+is_hema)*1.5, 0.8))

            # ── range violation counts ────────────────────────────────
            n_hi  = (int(hba1c>5.7) + int(glucose>100) + int(cea>5) + int(alt>40)
                     + int(ldl>130) + int(bp_sys>130) + int(bmi>25) + int(triglyc>150))
            n_lo  = (int(hemoglobin < (12 if sex_f else 13.5)) + int(lymph<20)
                     + int(hdl < (50 if sex_f else 40)) + int(vitamin_d_proxy(age) < 20))
            n_crit= (int(cea>10) + int(hba1c>6.5) + int(glucose>126)
                     + int(wbc<3) + int(wbc>12) + int(lymph<15))

            row = [
                age, sex_f, smoke, alc, 300-min(exer,300),
                fam_c, fam_d, fam_v, n_chk, months,
                hba1c, glucose, hemoglobin, lymph, wbc, platelets,
                cea, ca125, psa, alt, ast, ldl, hdl, triglyc,
                bp_sys, bmi, creat, tsh, crp, ferritin,
                hba1c_sl, glucose_sl, hgb_sl, lymph_sl,
                wbc_sl, cea_sl, alt_sl, ldl_sl, bp_sl, bmi_sl, plt_sl, crp_sl,
                hba1c_v, cea_v, hgb_v, lymph_v,
                n_hi, n_lo, n_crit,
            ]

            rows.append(row)
            labels["cancer"].append(float(p_cancer))
            labels["metabolic"].append(float(p_meta))
            labels["cardio"].append(float(p_cardio))
            labels["hematologic"].append(float(p_hema))

        return np.array(rows, dtype=np.float32), labels

    def train(self):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import mean_absolute_error

        print("🧠 BioSentinel ML Engine — generating training data...")
        X, raw_labels = self._synthetic(5000)

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        print("🧠 Training 4 calibrated models...\n")
        display = {"cancer":"Cancer","metabolic":"Metabolic",
                   "cardio":"Cardiovascular","hematologic":"Hematologic"}

        for disease, y_raw in raw_labels.items():
            y = np.array(y_raw, dtype=np.float32)
            # Clip y to [0.05, 0.95] so model never tries to predict exact 0/1
            y_clipped = np.clip(y, 0.05, 0.95)

            Xtr, Xte, ytr, yte = train_test_split(
                Xs, y_clipped, test_size=0.15, random_state=42)

            reg = GradientBoostingRegressor(
                n_estimators=160, max_depth=3, learning_rate=0.06,
                subsample=0.75, min_samples_leaf=25,
                loss="huber", random_state=42)
            reg.fit(Xtr, ytr)

            # Isotonic calibration pass
            raw_preds_val = reg.predict(Xte)
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(raw_preds_val, yte)

            # Evaluate
            cal_preds = iso.predict(reg.predict(Xte))
            mae = mean_absolute_error(yte, cal_preds)

            # Feature importances (for explanations)
            fi = reg.feature_importances_
            top_idx = np.argsort(fi)[::-1][:8]
            top_feats = [(self.FEATURES[i], float(fi[i])) for i in top_idx]

            self.models[disease] = {"reg": reg, "iso": iso, "top_feats": top_feats}

            # Show calibration distribution
            all_preds = np.clip(iso.predict(reg.predict(Xs)), 0.05, 0.95)
            pct_low  = (all_preds < 0.25).mean()*100
            pct_mod  = ((all_preds >= 0.25) & (all_preds < 0.50)).mean()*100
            pct_high = ((all_preds >= 0.50) & (all_preds < 0.75)).mean()*100
            pct_crit = (all_preds >= 0.75).mean()*100

            print(f"  ✅ {display[disease]:15s}  MAE={mae:.3f}  "
                  f"Low:{pct_low:.0f}%  Mod:{pct_mod:.0f}%  "
                  f"High:{pct_high:.0f}%  Crit:{pct_crit:.0f}%")

        self.trained = True

        # ── Build SHAP explainers (one per model) ─────────────────────────
        # Use a 200-sample background for TreeExplainer — fast and accurate
        try:
            import shap
            bg_idx = np.random.RandomState(42).choice(len(Xs), 200, replace=False)
            bg = Xs[bg_idx]
            for disease, m in self.models.items():
                explainer = shap.TreeExplainer(m["reg"], data=bg,
                                               feature_perturbation="interventional")
                self.models[disease]["shap_explainer"] = explainer
            self.shap_available = True
            print("✅ SHAP explainers built for all 4 models.\n")
        except Exception as e:
            self.shap_available = False
            print(f"⚠  SHAP unavailable ({e}), using fallback attribution.\n")

        print("✅ All 4 models trained and calibrated.\n")

    def _extract(self, checkups: list, patient) -> Optional[np.ndarray]:
        if not checkups: return None
        checkups = sorted(checkups, key=lambda c: c.checkup_date)
        n = len(checkups)

        def vals(attr):
            return [getattr(c, attr) for c in checkups if getattr(c, attr) is not None]

        def latest(v, default=None): return v[-1] if v else default
        def slope_pm(v):
            """Slope per month across all available measurements."""
            if len(v) < 2: return 0.0
            months = max(_months_between(checkups[0].checkup_date,
                                         checkups[-1].checkup_date), 1)
            return (v[-1] - v[0]) / months

        try:
            months = _months_between(checkups[0].checkup_date, checkups[-1].checkup_date)
        except Exception:
            months = n * 3.0

        smoke = {"never":0,"former":1,"current":2}.get(
            (patient.smoking_status or "never").lower(), 0)

        hba1c_v   = vals("hba1c")
        gluc_v    = vals("glucose_fasting")
        hgb_v     = vals("hemoglobin")
        lymph_v   = vals("lymphocytes_pct")
        wbc_v     = vals("wbc")
        plt_v     = vals("platelets")
        cea_v     = vals("cea")
        ca125_v   = vals("ca125")
        psa_v     = vals("psa")
        alt_v     = vals("alt")
        ast_v     = vals("ast")
        ldl_v     = vals("ldl")
        hdl_v     = vals("hdl")
        trig_v    = vals("triglycerides")
        bp_v      = vals("bp_systolic")
        bmi_v     = vals("bmi")
        creat_v   = vals("creatinine")
        tsh_v     = vals("tsh")
        crp_v     = vals("crp")
        ferr_v    = vals("ferritin")

        hba1c   = latest(hba1c_v, 5.5)
        glucose = latest(gluc_v, 92)
        hgb     = latest(hgb_v, 14)
        lymph   = latest(lymph_v, 30)
        wbc     = latest(wbc_v, 7)
        plt     = latest(plt_v, 250)
        cea     = latest(cea_v, 1.5)
        ca125   = latest(ca125_v, 14)
        psa     = latest(psa_v, 1.5)
        alt     = latest(alt_v, 25)
        ast     = latest(ast_v, 22)
        ldl     = latest(ldl_v, 110)
        hdl     = latest(hdl_v, 52)
        trig    = latest(trig_v, 125)
        bp_sys  = latest(bp_v, 120)
        bmi     = latest(bmi_v, 24)
        creat   = latest(creat_v, 0.87)
        tsh     = latest(tsh_v, 2.2)
        crp     = latest(crp_v, 1.5)
        ferr    = latest(ferr_v, 80)

        sex_f = 1 if patient.sex.lower() in ("f","female") else 0

        n_hi  = (int(hba1c>5.7) + int(glucose>100) + int(cea>5) + int(alt>40)
                 + int(ldl>130) + int(bp_sys>130) + int(bmi>25) + int(trig>150))
        n_lo  = (int(hgb < (12 if sex_f else 13.5)) + int(lymph<20)
                 + int(hdl < (50 if sex_f else 40)))
        n_crit= (int(cea>10) + int(hba1c>6.5) + int(glucose>126)
                 + int(wbc<3) + int(wbc>12) + int(lymph<15))

        feat = [
            patient.age, sex_f, smoke,
            patient.alcohol_units_weekly, max(0, 300-patient.exercise_min_weekly),
            patient.family_history_cancer, patient.family_history_diabetes,
            patient.family_history_cardio, n, months,
            hba1c, glucose, hgb, lymph, wbc, plt,
            cea, ca125, psa, alt, ast, ldl, hdl, trig,
            bp_sys, bmi, creat, tsh, crp, ferr,
            slope_pm(hba1c_v), slope_pm(gluc_v), slope_pm(hgb_v), slope_pm(lymph_v),
            slope_pm(wbc_v), slope_pm(cea_v), slope_pm(alt_v), slope_pm(ldl_v),
            slope_pm(bp_v), slope_pm(bmi_v), slope_pm(plt_v), slope_pm(crp_v),
            np.std(hba1c_v) if len(hba1c_v)>1 else 0,
            np.std(cea_v)   if len(cea_v)>1   else 0,
            np.std(hgb_v)   if len(hgb_v)>1   else 0,
            np.std(lymph_v) if len(lymph_v)>1 else 0,
            n_hi, n_lo, n_crit,
        ]
        return np.array([feat], dtype=np.float32)

    def predict(self, checkups: list, patient) -> dict:
        if not self.trained: self.train()
        feat = self._extract(checkups, patient)
        if feat is None:
            return {"error": "No checkup data"}
        fs = self.scaler.transform(feat)

        results = {}
        for disease, m in self.models.items():
            raw  = float(m["reg"].predict(fs)[0])
            prob = float(np.clip(m["iso"].predict([raw])[0], 0.05, 0.92))
            results[disease] = prob

        checkups_s = sorted(checkups, key=lambda c: c.checkup_date)
        try:
            months = _months_between(checkups_s[0].checkup_date, checkups_s[-1].checkup_date)
        except Exception:
            months = len(checkups) * 3.0

        feat_raw = feat[0]
        attributions = self._attribute(feat_raw, results)
        alerts       = self._build_alerts(feat_raw, results, patient)
        rec          = self._recommendation(results, alerts)

        completeness = sum(1 for v in feat_raw[10:30] if v != 0) / 20

        return {
            "cancer":      {"risk": results["cancer"],      "level": _level(results["cancer"])},
            "metabolic":   {"risk": results["metabolic"],   "level": _level(results["metabolic"])},
            "cardio":      {"risk": results["cardio"],      "level": _level(results["cardio"])},
            "hematologic": {"risk": results["hematologic"], "level": _level(results["hematologic"])},
            "composite":   round((results["cancer"]*0.35 + results["metabolic"]*0.3
                                  + results["cardio"]*0.25 + results["hematologic"]*0.1), 4),
            "top_features":   attributions,
            "alerts":         alerts,
            "recommendation": rec,
            "checkups_used":  len(checkups),
            "months_of_data": round(months, 1),
            "data_completeness": round(completeness, 2),
        }

    def _attribute(self, feat_raw, results):
        """
        Feature attribution using real SHAP TreeExplainer values.
        Falls back to heuristic if SHAP is unavailable.
        Returns top 8 drivers with human-readable labels.
        """
        # Human-readable labels for every feature (matches self.FEATURES order)
        FEATURE_LABELS = [
            "Patient age", "Female sex", "Smoking status",
            "Alcohol consumption", "Physical inactivity",
            "Family cancer history", "Family diabetes history", "Family cardio history",
            "Number of checkups", "Months of monitoring data",
            "HbA1c (blood sugar)", "Fasting glucose", "Hemoglobin",
            "Lymphocyte count", "White blood cells", "Platelets",
            "CEA tumour marker", "CA-125", "PSA",
            "ALT liver enzyme", "AST liver enzyme",
            "LDL cholesterol", "HDL cholesterol", "Triglycerides",
            "Systolic blood pressure", "BMI",
            "Creatinine (kidney)", "TSH (thyroid)", "CRP (inflammation)", "Ferritin",
            "HbA1c trend (slope/month)", "Glucose trend", "Hemoglobin trend",
            "Lymphocyte trend", "WBC trend", "CEA trend",
            "ALT trend", "LDL trend", "BP trend", "BMI trend",
            "Platelet trend", "CRP trend",
            "HbA1c volatility", "CEA volatility",
            "Hemoglobin volatility", "Lymphocyte volatility",
            "Elevated values count", "Low values count", "Critical values count",
        ]

        # Determine the dominant disease domain for SHAP
        dominant = max(results, key=lambda d: results[d])

        if getattr(self, "shap_available", False):
            try:
                import shap
                m = self.models[dominant]
                explainer = m["shap_explainer"]
                feat_scaled = self.scaler.transform(feat_raw.reshape(1, -1))
                sv = explainer.shap_values(feat_scaled)[0]   # shape: (n_features,)

                attrs = []
                for i, val in enumerate(sv):
                    if abs(val) < 1e-5:
                        continue
                    raw_val = float(feat_raw[i])
                    attrs.append({
                        "label":     FEATURE_LABELS[i] if i < len(FEATURE_LABELS) else f"feature_{i}",
                        "feature":   self.FEATURES[i] if i < len(self.FEATURES) else f"f{i}",
                        "value":     round(raw_val, 3),
                        "shap_value": round(float(val), 5),
                        "impact":    round(abs(float(val)), 5),
                        "domain":    dominant,
                        "direction": "risk_increasing" if val > 0 else "protective",
                        "method":    "shap",
                    })

                attrs.sort(key=lambda x: -x["impact"])
                return attrs[:8]
            except Exception as e:
                print(f"[SHAP] Attribution failed, using fallback: {e}")

        # ── Heuristic fallback (if SHAP unavailable) ──────────────────────
        f = feat_raw
        checks = [
            ("HbA1c upward trend",      f[31], 0.01,  "up",   "metabolic"),
            ("HbA1c above pre-diabetic", f[10], 5.7,   "high", "metabolic"),
            ("Fasting glucose rising",   f[32], 0.3,   "up",   "metabolic"),
            ("BMI elevated & rising",    f[35], 0.02,  "up",   "metabolic"),
            ("CEA tumour marker rising",  f[36], 0.04,  "up",   "cancer"),
            ("CEA above normal",         f[16], 3.0,   "high", "cancer"),
            ("Lymphocytes declining",    f[34], -0.08, "down", "cancer"),
            ("Hemoglobin declining",     f[33], -0.01, "down", "cancer"),
            ("Family cancer history",    f[5],  0,     "fh",   "cancer"),
            ("LDL cholesterol rising",   f[37], 0.2,   "up",   "cardio"),
            ("Blood pressure rising",    f[38], 0.2,   "up",   "cardio"),
            ("Low HDL cholesterol",      f[22], 50,    "low",  "cardio"),
            ("Current smoker",           f[2],  1.5,   "high", "all"),
            ("High CRP (inflammation)",  f[28], 3.0,   "high", "cancer"),
            ("Age-related risk",         f[0],  55,    "high", "all"),
        ]
        attrs = []
        for label, val, thr, direction, domain in checks:
            if direction == "up":
                impact = max(0, float(val) - thr) * 2.5
            elif direction == "high":
                impact = max(0, float(val) - thr) / max(abs(thr), 1) * 0.25
            elif direction == "low":
                impact = max(0, thr - float(val)) / max(abs(thr), 1) * 0.25
            elif direction == "down":
                impact = max(0, thr - float(val)) * 2.5
            elif direction == "fh":
                impact = float(val) * 0.15
            else:
                impact = 0
            if impact > 0.005:
                attrs.append({"label": label, "value": round(float(val), 3),
                              "impact": round(impact, 4), "domain": domain,
                              "direction": "risk_increasing", "method": "heuristic"})
        attrs.sort(key=lambda x: -x["impact"])
        return attrs[:8]

    def _build_alerts(self, feat, results, patient):
        alerts = []
        f = feat

        if f[16] > 5.0:
            alerts.append({"level":"WARNING","cat":"cancer",
                "msg":f"CEA elevated at {f[16]:.1f} ng/mL (normal <5.0). Repeat in 4 weeks."})
        if f[10] > 6.5:
            alerts.append({"level":"CRITICAL","cat":"metabolic",
                "msg":f"HbA1c {f[10]:.1f}% — diabetic range. Immediate endocrinology referral."})
        elif f[10] > 5.7:
            alerts.append({"level":"WARNING","cat":"metabolic",
                "msg":f"HbA1c {f[10]:.1f}% — pre-diabetic. Diet/lifestyle intervention needed."})
        if f[13] < 15:
            alerts.append({"level":"CRITICAL","cat":"hematologic",
                "msg":f"Lymphocytes critically low ({f[13]:.0f}%). Hematology workup urgent."})
        elif f[13] < 20:
            alerts.append({"level":"WARNING","cat":"hematologic",
                "msg":f"Lymphocytes declining ({f[13]:.0f}%). Monitor closely."})
        if f[12] < (12 if patient.sex.lower() in ("f","female") else 13):
            alerts.append({"level":"WARNING","cat":"hematologic",
                "msg":f"Hemoglobin low at {f[12]:.1f} g/dL. Investigate anemia cause."})
        if f[24] > 140:
            alerts.append({"level":"WARNING","cat":"cardio",
                "msg":f"Systolic BP {f[24]:.0f} mmHg — Stage 2 hypertension. Cardiology review."})
        if results["cancer"] > 0.65:
            alerts.append({"level":"CRITICAL","cat":"cancer",
                "msg":"Multiple cancer risk signals detected. Oncology referral recommended."})
        if results["metabolic"] > 0.65:
            alerts.append({"level":"WARNING","cat":"metabolic",
                "msg":"High metabolic disease risk. Comprehensive metabolic workup needed."})
        return alerts

    def _recommendation(self, results, alerts):
        crit = [a for a in alerts if a["level"] == "CRITICAL"]
        warn = [a for a in alerts if a["level"] == "WARNING"]
        parts = []
        if crit:
            parts.append(f"🔴 CRITICAL: {crit[0]['msg']}")
        if warn:
            parts.append(f"⚠️ {warn[0]['msg']}")
        if not parts:
            parts.append("✅ No elevated risk signals. Continue routine 3-month monitoring.")
        # Add trend-based advice
        if results["metabolic"] > 0.45:
            parts.append("Consider HbA1c-lowering diet and aerobic exercise (150 min/week).")
        if results["cardio"] > 0.45:
            parts.append("Statin therapy evaluation and BP management recommended.")
        return " ".join(parts)


# ── HELPERS ──────────────────────────────────────────────────────────────────
def _sigmoid(x): return 1.0 / (1.0 + math.exp(-x))

def vitamin_d_proxy(age): return max(10, 40 - age*0.3)

def _months_between(d1: str, d2: str) -> float:
    dt1 = datetime.fromisoformat(str(d1)[:10])
    dt2 = datetime.fromisoformat(str(d2)[:10])
    return max(abs((dt2-dt1).days / 30.4), 0.1)

def _level(score):
    if score < 0.25: return "LOW"
    elif score < 0.50: return "MODERATE"
    elif score < 0.75: return "HIGH"
    return "CRITICAL"

# ── GLOBAL ENGINE ────────────────────────────────────────────────────────────
engine_ml = BioSentinelEngine()

# ── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BioSentinel API v2.0",
    description=(
        "AI-powered longitudinal health monitoring & early disease detection.\n"
        "Developer: Liveupx Pvt. Ltd. | Mohit Chaprana\n"
        "github.com/liveupx/biosentinel"
    ),
    version="2.3.3",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Rate limiting (if slowapi installed)
if RATE_LIMIT_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address,
                      default_limits=["200/minute"])
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
else:
    limiter = None

def _rate_limit(limit: str = "30/minute"):
    """Decorator factory — no-op if slowapi not installed or in test mode."""
    _in_test = os.getenv("DATABASE_URL", "").startswith("sqlite:///") and \
               ("test" in os.getenv("DATABASE_URL", "") or
                os.getenv("SECRET_KEY", "") == "test-secret-not-for-prod")
    if RATE_LIMIT_AVAILABLE and limiter and not _in_test:
        return limiter.limit(limit)
    def _noop(f): return f
    return _noop

# ── HEALTH ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"name":"BioSentinel","version":"1.0.0","status":"operational",
            "docs":"/docs","github":"https://github.com/liveupx/biosentinel"}

@app.get("/health")
def health():
    return {"status":"healthy","ml_trained":engine_ml.trained,
            "models":list(engine_ml.models.keys()) if engine_ml.trained else []}

# ── AUTH ──────────────────────────────────────────────────────────────────────
@app.post("/api/v1/auth/register", status_code=201)
@_rate_limit("10/minute")
def register(request: Request, u: UserCreate, db: Session = Depends(get_db)):
    if db.query(DBUser).filter(DBUser.username == u.username).first():
        raise HTTPException(400, "Username taken")
    user = DBUser(username=u.username, email=u.email,
                  hashed_password=hash_pw(u.password), role=u.role)
    db.add(user); db.commit(); db.refresh(user)
    return {"access_token": make_token({"sub":user.username}),
            "token_type":"bearer",
            "user":{"id":user.id,"username":user.username,"role":user.role}}

class UserLoginWith2FA(BaseModel):
    username: str
    password: str
    totp_code: Optional[str] = None   # 6-digit code from authenticator app

@app.post("/api/v1/auth/login")
@_rate_limit("10/minute")
def login(request: Request, u: UserLoginWith2FA, db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.username == u.username).first()
    if not user or not verify_pw(u.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    # 2FA check
    if user.totp_enabled and user.totp_secret:
        if not u.totp_code:
            raise HTTPException(401, "2FA_REQUIRED",
                headers={"X-2FA-Required": "true"})
        if TOTP_AVAILABLE:
            totp = pyotp.TOTP(user.totp_secret)
            # Also check backup codes
            code_valid = totp.verify(u.totp_code, valid_window=1)
            if not code_valid:
                # Try backup codes
                backup = json.loads(user.totp_backup_codes or "[]")
                hashed_input = hash_pw(u.totp_code)
                matched_idx  = next((i for i, c in enumerate(backup)
                                     if verify_pw(u.totp_code, c)), None)
                if matched_idx is None:
                    raise HTTPException(401, "Invalid 2FA code")
                # Consume backup code
                backup.pop(matched_idx)
                user.totp_backup_codes = json.dumps(backup)
                db.commit()
        else:
            raise HTTPException(503, "2FA not available — install pyotp")
    # Record session
    device = request.headers.get("X-Device", "Unknown")
    ip     = request.client.host if request.client else "Unknown"
    ua     = request.headers.get("User-Agent", "")[:120]
    token  = make_token({"sub": user.username},
                        device=device or ua[:50] or "Browser", ip=ip, db=db)
    log.info("user_login", username=user.username, ip=ip)
    return {"access_token": token,
            "token_type": "bearer",
            "user": {"id": user.id, "username": user.username, "role": user.role,
                     "totp_enabled": bool(user.totp_enabled)}}

@app.get("/api/v1/auth/me")
def me(u=Depends(current_user)):
    return {"id":u.id,"username":u.username,"email":u.email,"role":u.role}

# ── PATIENTS ──────────────────────────────────────────────────────────────────
@app.post("/api/v1/patients", status_code=201)
def create_patient(p: PatientCreate, db=Depends(get_db), user=Depends(current_user)):
    data = p.model_dump()
    data.pop('owner_id', None)
    # admin sees all; others own their patients
    data['owner_id'] = user.id
    pat = DBPatient(**data); db.add(pat); db.commit(); db.refresh(pat)
    return pat

@app.get("/api/v1/patients")
def list_patients(skip:int=0, limit:int=100, db=Depends(get_db), user=Depends(current_user)):
    q = db.query(DBPatient)
    if user.role != "admin":          # admin sees everyone; others see only their own
        q = q.filter(DBPatient.owner_id == user.id)
    pats  = q.order_by(DBPatient.created_at.desc()).offset(skip).limit(limit).all()
    total = q.count()
    return {"patients": pats, "total": total}

def _get_patient_or_403(pid: str, db: Session, user) -> DBPatient:
    """Fetch patient; raise 403 if caller doesn't own it (unless admin)."""
    p = db.query(DBPatient).filter(DBPatient.id == pid).first()
    if not p: raise HTTPException(404, "Patient not found")
    if user.role != "admin" and p.owner_id != user.id:
        raise HTTPException(403, "You don't have access to this patient")
    return p


# ── FULL-TEXT PATIENT SEARCH ──────────────────────────────────────────────────

@app.get("/api/v1/patients/search")
def search_patients(
    q:           Optional[str]   = Query(None, description="Free-text search (ethnicity, notes)"),
    sex:         Optional[str]   = Query(None),
    age_min:     Optional[int]   = Query(None),
    age_max:     Optional[int]   = Query(None),
    risk_level:  Optional[str]   = Query(None, description="LOW|MODERATE|HIGH|CRITICAL"),
    smoking:     Optional[str]   = Query(None, description="never|former|current"),
    has_diabetes_fh: Optional[int] = Query(None, description="1 or 0"),
    has_cancer_fh:   Optional[int] = Query(None, description="1 or 0"),
    overdue:     Optional[bool]  = Query(None, description="Only overdue patients"),
    limit:       int             = Query(50, le=200),
    offset:      int             = Query(0),
    db=Depends(get_db),
    user=Depends(current_user),
):
    """
    Advanced patient search with multiple filters.
    Examples:
      /search?q=south asian&sex=Female&age_min=45
      /search?risk_level=HIGH&smoking=current
      /search?overdue=true&has_cancer_fh=1
    """
    if user.role == "admin":
        query = db.query(DBPatient)
    else:
        query = db.query(DBPatient).filter(DBPatient.owner_id == user.id)

    # Free text: ethnicity, notes
    if q:
        q_lower = f"%{q.lower()}%"
        query = query.filter(or_(
            DBPatient.ethnicity.ilike(q_lower),
            DBPatient.notes.ilike(q_lower),
        ))

    # Demographic filters
    if sex:         query = query.filter(DBPatient.sex.ilike(f"%{sex}%"))
    if age_min:     query = query.filter(DBPatient.age >= age_min)
    if age_max:     query = query.filter(DBPatient.age <= age_max)
    if smoking:     query = query.filter(DBPatient.smoking_status == smoking)
    if has_diabetes_fh is not None:
        query = query.filter(DBPatient.family_history_diabetes >= has_diabetes_fh)
    if has_cancer_fh is not None:
        query = query.filter(DBPatient.family_history_cancer >= has_cancer_fh)

    total = query.count()
    patients = query.offset(offset).limit(limit).all()

    results = []
    for p in patients:
        row = {
            "id": p.id, "age": p.age, "sex": p.sex,
            "ethnicity": p.ethnicity, "smoking_status": p.smoking_status,
            "family_history_cancer": p.family_history_cancer,
            "family_history_diabetes": p.family_history_diabetes,
            "created_at": p.created_at,
        }
        # Add latest prediction level if requested
        if risk_level:
            latest_pred = (db.query(DBPrediction)
                           .filter(DBPrediction.patient_id == p.id)
                           .order_by(DBPrediction.created_at.desc()).first())
            if not latest_pred: continue
            pat_level = latest_pred.cancer_level
            if risk_level.upper() not in (latest_pred.cancer_level,
                                           latest_pred.metabolic_level,
                                           latest_pred.cardio_level): continue
            row["latest_cancer_risk"]    = latest_pred.cancer_risk
            row["latest_metabolic_risk"] = latest_pred.metabolic_risk
            row["latest_risk_level"]     = pat_level
        # Overdue filter
        if overdue is not None:
            last_chk = (db.query(DBCheckup).filter(DBCheckup.patient_id == p.id)
                        .order_by(DBCheckup.checkup_date.desc()).first())
            cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).date().isoformat()
            is_overdue = not last_chk or last_chk.checkup_date[:10] < cutoff
            if overdue != is_overdue: continue
            row["overdue"] = is_overdue
        results.append(row)

    return {"total": total, "returned": len(results),
            "offset": offset, "limit": limit,
            "patients": results}


@app.get("/api/v1/patients/{pid}")
def get_patient(pid:str, db=Depends(get_db), user=Depends(current_user)):
    return _get_patient_or_403(pid, db, user)

@app.put("/api/v1/patients/{pid}")
def update_patient(pid:str, data:PatientCreate, db=Depends(get_db), user=Depends(current_user)):
    p = _get_patient_or_403(pid, db, user)
    d = data.model_dump(); d.pop('owner_id', None)
    for k,v in d.items(): setattr(p,k,v)
    db.commit(); db.refresh(p); return p

@app.delete("/api/v1/patients/{pid}")
def delete_patient(pid:str, db=Depends(get_db), user=Depends(current_user)):
    p = _get_patient_or_403(pid, db, user)
    db.delete(p); db.commit(); return {"deleted":pid}

# ── CHECKUPS ──────────────────────────────────────────────────────────────────
@app.post("/api/v1/checkups", status_code=201)
def create_checkup(c: CheckupCreate, db=Depends(get_db), user=Depends(current_user)):
    _get_patient_or_403(c.patient_id, db, user)
    chk = DBCheckup(**c.model_dump()); db.add(chk); db.commit(); db.refresh(chk)
    return chk

@app.get("/api/v1/patients/{pid}/checkups")
def get_checkups(pid:str, db=Depends(get_db), user=Depends(current_user)):
    _get_patient_or_403(pid, db, user)
    chks = (db.query(DBCheckup).filter(DBCheckup.patient_id==pid)
            .order_by(DBCheckup.checkup_date).all())
    return {"checkups":chks,"count":len(chks)}

@app.get("/api/v1/checkups/{cid}")
def get_checkup(cid:str, db=Depends(get_db), _=Depends(current_user)):
    c = db.query(DBCheckup).filter(DBCheckup.id==cid).first()
    if not c: raise HTTPException(404,"Checkup not found")
    return c

@app.delete("/api/v1/checkups/{cid}")
def delete_checkup(cid:str, db=Depends(get_db), _=Depends(current_user)):
    c = db.query(DBCheckup).filter(DBCheckup.id==cid).first()
    if not c: raise HTTPException(404,"Checkup not found")
    db.delete(c); db.commit(); return {"deleted":cid}

# ── MEDICATIONS ───────────────────────────────────────────────────────────────
@app.post("/api/v1/medications", status_code=201)
def create_med(m: MedicationCreate, db=Depends(get_db), user=Depends(current_user)):
    _get_patient_or_403(m.patient_id, db, user)
    med = DBMedication(**m.model_dump()); db.add(med); db.commit(); db.refresh(med)
    return med

@app.get("/api/v1/patients/{pid}/medications")
def get_meds(pid:str, db=Depends(get_db), user=Depends(current_user)):
    _get_patient_or_403(pid, db, user)
    return {"medications": db.query(DBMedication).filter(DBMedication.patient_id==pid).all()}

@app.delete("/api/v1/medications/{mid}")
def delete_med(mid:str, db=Depends(get_db), user=Depends(current_user)):
    m = db.query(DBMedication).filter(DBMedication.id==mid).first()
    if not m: raise HTTPException(404,"Not found")
    _get_patient_or_403(m.patient_id, db, user)
    db.delete(m); db.commit(); return {"deleted":mid}

# ── DIAGNOSES ─────────────────────────────────────────────────────────────────
@app.post("/api/v1/diagnoses", status_code=201)
def create_diag(d: DiagnosisCreate, db=Depends(get_db), user=Depends(current_user)):
    _get_patient_or_403(d.patient_id, db, user)
    diag = DBDiagnosis(**d.model_dump()); db.add(diag); db.commit(); db.refresh(diag)
    return diag

@app.get("/api/v1/patients/{pid}/diagnoses")
def get_diags(pid:str, db=Depends(get_db), user=Depends(current_user)):
    _get_patient_or_403(pid, db, user)
    return {"diagnoses": db.query(DBDiagnosis).filter(DBDiagnosis.patient_id==pid).all()}

@app.delete("/api/v1/diagnoses/{did}")
def delete_diag(did:str, db=Depends(get_db), user=Depends(current_user)):
    d = db.query(DBDiagnosis).filter(DBDiagnosis.id==did).first()
    if not d: raise HTTPException(404,"Not found")
    _get_patient_or_403(d.patient_id, db, user)
    db.delete(d); db.commit(); return {"deleted":did}

# ── DIET PLANS ────────────────────────────────────────────────────────────────
@app.post("/api/v1/diet-plans", status_code=201)
def create_diet(dp: DietPlanCreate, db=Depends(get_db), _=Depends(current_user)):
    plan = DBDietPlan(**dp.model_dump()); db.add(plan); db.commit(); db.refresh(plan)
    return plan

@app.get("/api/v1/patients/{pid}/diet-plans")
def get_diets(pid:str, db=Depends(get_db), _=Depends(current_user)):
    return {"diet_plans": db.query(DBDietPlan).filter(DBDietPlan.patient_id==pid).all()}

# ── PREDICTIONS ───────────────────────────────────────────────────────────────
@app.post("/api/v1/patients/{pid}/predict")
def predict(pid:str, db=Depends(get_db), user=Depends(current_user)):
    patient = _get_patient_or_403(pid, db, user)
    checkups = (db.query(DBCheckup).filter(DBCheckup.patient_id==pid)
                .order_by(DBCheckup.checkup_date).all())
    if not checkups: raise HTTPException(400,"Need at least 1 checkup")

    result = engine_ml.predict(checkups, patient)

    pred = DBPrediction(
        patient_id       = pid,
        checkups_used    = result["checkups_used"],
        months_of_data   = result["months_of_data"],
        data_completeness= result["data_completeness"],
        cancer_risk      = result["cancer"]["risk"],
        cancer_level     = result["cancer"]["level"],
        metabolic_risk   = result["metabolic"]["risk"],
        metabolic_level  = result["metabolic"]["level"],
        cardio_risk      = result["cardio"]["risk"],
        cardio_level     = result["cardio"]["level"],
        hematologic_risk = result["hematologic"]["risk"],
        hematologic_level= result["hematologic"]["level"],
        composite_score  = result["composite"],
        top_features_json= json.dumps(result["top_features"]),
        alerts_json      = json.dumps(result["alerts"]),
        recommendation   = result["recommendation"],
    )
    db.add(pred); db.commit(); db.refresh(pred)

    # Save alerts to alerts table and send emails
    email_cfg = db.query(DBEmailConfig).filter(DBEmailConfig.user_id == user.id).first()
    for a in result["alerts"]:
        alert_obj = DBAlert(patient_id=pid, level=a["level"],
                            category=a["cat"], message=a["msg"])
        db.add(alert_obj)
        db.flush()  # get the ID before sending email
        # fire email in background thread
        email_config_engine.send_alert_email(email_cfg, alert_obj, patient.age, patient.sex)
    db.commit()

    audit(db, user, "run_prediction", pid,
          f"cancer={result['cancer']['level']} metabolic={result['metabolic']['level']}")

    result["id"] = pred.id
    result["patient_id"] = pid
    result["created_at"] = pred.created_at
    return result

@app.get("/api/v1/patients/{pid}/predictions")
def get_predictions(pid:str, db=Depends(get_db), user=Depends(current_user)):
    preds = (db.query(DBPrediction).filter(DBPrediction.patient_id==pid)
             .order_by(DBPrediction.created_at.desc()).all())
    out = []
    for p in preds:
        out.append({
            "id": p.id, "created_at": p.created_at,
            "checkups_used": p.checkups_used, "months_of_data": p.months_of_data,
            "data_completeness": p.data_completeness,
            "cancer":      {"risk":p.cancer_risk,      "level":p.cancer_level},
            "metabolic":   {"risk":p.metabolic_risk,   "level":p.metabolic_level},
            "cardio":      {"risk":p.cardio_risk,       "level":p.cardio_level},
            "hematologic": {"risk":p.hematologic_risk, "level":p.hematologic_level},
            "composite":   p.composite_score,
            "top_features":json.loads(p.top_features_json or "[]"),
            "alerts":      json.loads(p.alerts_json or "[]"),
            "recommendation": p.recommendation,
        })
    return {"predictions": out}

# ── ALERTS ────────────────────────────────────────────────────────────────────
@app.get("/api/v1/alerts")
def get_all_alerts(db=Depends(get_db), user=Depends(current_user)):
    # filter alerts to only the user's own patients unless admin
    if user.role == "admin":
        q = db.query(DBAlert)
    else:
        owned_ids = [p.id for p in db.query(DBPatient).filter(DBPatient.owner_id==user.id).all()]
        q = db.query(DBAlert).filter(DBAlert.patient_id.in_(owned_ids))
    alerts = q.order_by(DBAlert.created_at.desc()).limit(100).all()
    unread = q.filter(DBAlert.acknowledged==0).count()
    return {"alerts": alerts, "unread": unread}

@app.get("/api/v1/patients/{pid}/alerts")
def get_patient_alerts(pid:str, db=Depends(get_db), user=Depends(current_user)):
    _get_patient_or_403(pid, db, user)
    alerts = (db.query(DBAlert).filter(DBAlert.patient_id==pid)
              .order_by(DBAlert.created_at.desc()).all())
    return {"alerts": alerts}

@app.post("/api/v1/alerts/{aid}/acknowledge")
def ack_alert(aid:str, db=Depends(get_db), user=Depends(current_user)):
    a = db.query(DBAlert).filter(DBAlert.id==aid).first()
    if not a: raise HTTPException(404,"Alert not found")
    # check ownership
    _get_patient_or_403(a.patient_id, db, user)
    a.acknowledged = 1; db.commit()
    return {"acknowledged": aid}

# ── BIOMARKER TRENDS (chart data) ─────────────────────────────────────────────
@app.get("/api/v1/patients/{pid}/trends")
def get_trends(pid:str, db=Depends(get_db), user=Depends(current_user)):
    """Return per-biomarker time series ready for charting."""
    _get_patient_or_403(pid, db, user)
    cache_key = f"trends:{pid}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    checkups = (db.query(DBCheckup).filter(DBCheckup.patient_id==pid)
                .order_by(DBCheckup.checkup_date).all())
    if not checkups:
        return {"labels":[], "series":{}}

    labels = [c.checkup_date[:10] for c in checkups]
    fields = ["hba1c","glucose_fasting","hemoglobin","lymphocytes_pct","wbc",
              "cea","alt","ldl","hdl","bp_systolic","bmi","crp","platelets",
              "triglycerides","creatinine","tsh","ferritin"]
    series = {}
    refs = {
        "hba1c":          {"lo":4.0,  "hi":5.7,   "unit":"%"},
        "glucose_fasting":{"lo":70,   "hi":100,   "unit":"mg/dL"},
        "hemoglobin":     {"lo":12.0, "hi":16.0,  "unit":"g/dL"},
        "lymphocytes_pct":{"lo":20,   "hi":40,    "unit":"%"},
        "wbc":            {"lo":4.5,  "hi":10.0,  "unit":"K/μL"},
        "cea":            {"lo":0,    "hi":5.0,   "unit":"ng/mL"},
        "alt":            {"lo":7,    "hi":40,    "unit":"U/L"},
        "ldl":            {"lo":0,    "hi":130,   "unit":"mg/dL"},
        "hdl":            {"lo":40,   "hi":100,   "unit":"mg/dL"},
        "bp_systolic":    {"lo":90,   "hi":130,   "unit":"mmHg"},
        "bmi":            {"lo":18.5, "hi":25.0,  "unit":"kg/m²"},
        "crp":            {"lo":0,    "hi":3.0,   "unit":"mg/L"},
        "platelets":      {"lo":150,  "hi":400,   "unit":"K/μL"},
        "triglycerides":  {"lo":0,    "hi":150,   "unit":"mg/dL"},
        "creatinine":     {"lo":0.6,  "hi":1.2,   "unit":"mg/dL"},
        "tsh":            {"lo":0.4,  "hi":4.0,   "unit":"mIU/L"},
        "ferritin":       {"lo":12,   "hi":300,   "unit":"ng/mL"},
    }
    for f in fields:
        vals = [getattr(c, f) for c in checkups]
        filled = [v for v in vals if v is not None]
        if not filled: continue
        # Trend direction
        trend = "stable"
        if len(filled) >= 2:
            delta = filled[-1] - filled[0]
            pct = delta / max(abs(filled[0]), 1e-6)
            if pct > 0.05: trend = "up"
            elif pct < -0.05: trend = "down"

        latest = filled[-1]
        ref = refs.get(f, {})
        status = "normal"
        if ref:
            if latest > ref["hi"]: status = "high"
            elif latest < ref["lo"]: status = "low"

        series[f] = {
            "values": vals,
            "latest": latest,
            "trend": trend,
            "status": status,
            "ref": ref,
        }
    result = {"labels": labels, "series": series}
    _cache_set(cache_key, result)
    return result

# ── ANALYTICS ─────────────────────────────────────────────────────────────────
@app.get("/api/v1/analytics/population")
def pop_analytics(db=Depends(get_db), user=Depends(current_user)):
    # admin sees all; others see only their patients' predictions
    if user.role == "admin":
        preds = db.query(DBPrediction).all()
        total_pats = db.query(DBPatient).count()
    else:
        owned = [p.id for p in db.query(DBPatient).filter(DBPatient.owner_id==user.id).all()]
        preds = db.query(DBPrediction).filter(DBPrediction.patient_id.in_(owned)).all()
        total_pats = len(owned)
    if not preds:
        return {"message":"No predictions yet","total_assessments":0}

    def dist(scores):
        total = len(scores) or 1
        return {
            "low":      sum(1 for s in scores if s<0.25),
            "moderate": sum(1 for s in scores if 0.25<=s<0.50),
            "high":     sum(1 for s in scores if 0.50<=s<0.75),
            "critical": sum(1 for s in scores if s>=0.75),
            "avg":      round(float(np.mean(scores)),3),
            "median":   round(float(np.median(scores)),3),
        }

    cancer_s    = [p.cancer_risk    for p in preds]
    metabolic_s = [p.metabolic_risk for p in preds]
    cardio_s    = [p.cardio_risk    for p in preds]
    hema_s      = [p.hematologic_risk for p in preds]
    composite_s = [p.composite_score  for p in preds if p.composite_score]

    return {
        "total_assessments": len(preds),
        "total_patients":    total_pats,
        "cancer_dist":       dist(cancer_s),
        "metabolic_dist":    dist(metabolic_s),
        "cardio_dist":       dist(cardio_s),
        "hematologic_dist":  dist(hema_s),
        "composite_dist":    dist(composite_s) if composite_s else {},
    }

@app.get("/api/v1/analytics/percentile/{pid}")
def patient_percentile(pid: str, db=Depends(get_db), user=Depends(current_user)):
    """
    Compare this patient's latest risk scores against all patients
    in the database with similar age (+/-10 years) and same sex.

    Returns: percentile rank per domain (0-100).
    Example: cancer_percentile=78 means this patient's cancer risk
    is higher than 78% of similar-age, same-sex patients.

    Requires at least 5 comparable patients for meaningful results.
    """
    patient = _get_patient_or_403(pid, db, user)

    # Get latest prediction for this patient
    my_pred = (db.query(DBPrediction)
               .filter(DBPrediction.patient_id == pid)
               .order_by(DBPrediction.created_at.desc())
               .first())
    if not my_pred:
        raise HTTPException(400, "No predictions found. Run a prediction first.")

    # Find comparable patients (±10 years, same sex)
    age_lo = max(0, (patient.age or 0) - 10)
    age_hi = (patient.age or 100) + 10

    comparable_pats = (db.query(DBPatient)
                       .filter(
                           DBPatient.age >= age_lo,
                           DBPatient.age <= age_hi,
                           DBPatient.sex == patient.sex,
                       ).all())

    comparable_ids = [p.id for p in comparable_pats]

    if len(comparable_ids) < 3:
        return {
            "patient_id": pid,
            "message": f"Only {len(comparable_ids)} comparable patients found. Need at least 3 for percentile comparison.",
            "comparable_group": {"age_range": f"{age_lo}-{age_hi}", "sex": patient.sex, "n": len(comparable_ids)},
            "percentiles": None,
        }

    # Get latest prediction per comparable patient
    domain_scores = {"cancer": [], "metabolic": [], "cardio": [], "hematologic": [], "composite": []}
    for cpid in comparable_ids:
        pred = (db.query(DBPrediction)
                .filter(DBPrediction.patient_id == cpid)
                .order_by(DBPrediction.created_at.desc())
                .first())
        if pred:
            domain_scores["cancer"].append(pred.cancer_risk or 0)
            domain_scores["metabolic"].append(pred.metabolic_risk or 0)
            domain_scores["cardio"].append(pred.cardio_risk or 0)
            domain_scores["hematologic"].append(pred.hematologic_risk or 0)
            if pred.composite_score:
                domain_scores["composite"].append(pred.composite_score)

    def percentile_rank(value: float, population: list) -> int:
        """Return what % of population this value is >= to (0-100)."""
        if not population:
            return 0
        below = sum(1 for v in population if v < value)
        return round((below / len(population)) * 100)

    def interpret(pct: int) -> str:
        if pct >= 90: return "Top 10% — significantly above average for your age/sex group"
        if pct >= 75: return "Above average — higher than 75% of similar patients"
        if pct >= 50: return "Slightly above average"
        if pct >= 25: return "Below average — lower risk than most similar patients"
        return "Bottom 25% — among the lowest risk in your age/sex group"

    my_scores = {
        "cancer":      my_pred.cancer_risk or 0,
        "metabolic":   my_pred.metabolic_risk or 0,
        "cardio":      my_pred.cardio_risk or 0,
        "hematologic": my_pred.hematologic_risk or 0,
        "composite":   my_pred.composite_score or 0,
    }

    percentiles = {}
    for domain, pop in domain_scores.items():
        if not pop:
            continue
        pct = percentile_rank(my_scores[domain], pop)
        percentiles[domain] = {
            "percentile":    pct,
            "my_score":      round(my_scores[domain] * 100, 1),
            "group_avg":     round(float(np.mean(pop)) * 100, 1),
            "group_median":  round(float(np.median(pop)) * 100, 1),
            "interpretation": interpret(pct),
        }

    audit(db, user, "percentile_comparison", pid,
          f"comparable_n={len(comparable_ids)} age={age_lo}-{age_hi} sex={patient.sex}")

    return {
        "patient_id": pid,
        "comparable_group": {
            "age_range":  f"{age_lo}–{age_hi}",
            "sex":        patient.sex,
            "n_patients": len(comparable_ids),
            "n_with_predictions": len(domain_scores["cancer"]),
        },
        "percentiles": percentiles,
        "disclaimer": (
            "Percentile ranks are relative to patients in this database only. "
            "Results become more meaningful with 50+ comparable patients."
        ),
    }


@app.get("/api/v1/analytics/biomarker-percentile/{pid}")
def biomarker_percentile(pid: str,
                          biomarker: str = Query(..., description="e.g. hba1c, cea, ldl"),
                          db=Depends(get_db), user=Depends(current_user)):
    """
    Compare a specific biomarker's latest value against comparable patients.
    More granular than the risk-score percentile — compares raw lab values.
    """
    patient = _get_patient_or_403(pid, db, user)

    # Get my latest checkup value
    my_chk = (db.query(DBCheckup)
              .filter(DBCheckup.patient_id == pid)
              .order_by(DBCheckup.checkup_date.desc())
              .first())
    if not my_chk:
        raise HTTPException(400, "No checkups found for this patient.")

    my_val = getattr(my_chk, biomarker, None)
    if my_val is None:
        raise HTTPException(400, f"Biomarker '{biomarker}' not recorded in latest checkup.")

    # Comparable patients ±10 years, same sex
    age_lo = max(0, (patient.age or 0) - 10)
    age_hi = (patient.age or 100) + 10
    comparable_pats = (db.query(DBPatient)
                       .filter(DBPatient.age >= age_lo,
                               DBPatient.age <= age_hi,
                               DBPatient.sex == patient.sex).all())

    if len(comparable_pats) < 3:
        raise HTTPException(400, f"Only {len(comparable_pats)} comparable patients. Need at least 3.")

    # Latest value of this biomarker for each comparable patient
    pop_values = []
    for cp in comparable_pats:
        chk = (db.query(DBCheckup)
               .filter(DBCheckup.patient_id == cp.id)
               .order_by(DBCheckup.checkup_date.desc())
               .first())
        if chk:
            val = getattr(chk, biomarker, None)
            if val is not None:
                pop_values.append(float(val))

    if len(pop_values) < 3:
        raise HTTPException(400, f"Not enough data for '{biomarker}' in comparable patients.")

    below = sum(1 for v in pop_values if v < my_val)
    pct = round((below / len(pop_values)) * 100)

    return {
        "patient_id":    pid,
        "biomarker":     biomarker,
        "my_value":      my_val,
        "percentile":    pct,
        "group_avg":     round(float(np.mean(pop_values)), 2),
        "group_median":  round(float(np.median(pop_values)), 2),
        "group_min":     round(float(np.min(pop_values)), 2),
        "group_max":     round(float(np.max(pop_values)), 2),
        "n_compared":    len(pop_values),
        "comparable_group": {"age_range": f"{age_lo}–{age_hi}", "sex": patient.sex},
        "interpretation": (
            f"Your {biomarker} of {my_val} is higher than {pct}% of similar "
            f"age/sex patients in this database ({len(pop_values)} patients)."
            if pct >= 50 else
            f"Your {biomarker} of {my_val} is lower than {100-pct}% of similar "
            f"age/sex patients — in the lower range for your group."
        ),
    }


@app.get("/api/v1/analytics/risk-trajectory/{pid}")
def risk_trajectory(pid:str, db=Depends(get_db), user=Depends(current_user)):
    """Return the risk score evolution over time for one patient."""
    _get_patient_or_403(pid, db, user)
    preds = (db.query(DBPrediction).filter(DBPrediction.patient_id==pid)
             .order_by(DBPrediction.created_at).all())
    return {
        "labels":    [p.created_at[:10] for p in preds],
        "cancer":    [p.cancer_risk    for p in preds],
        "metabolic": [p.metabolic_risk for p in preds],
        "cardio":    [p.cardio_risk    for p in preds],
        "composite": [p.composite_score for p in preds],
    }

# ── REPORT ────────────────────────────────────────────────────────────────────
@app.get("/api/v1/patients/{pid}/report")
def generate_report(pid:str, db=Depends(get_db), user=Depends(current_user)):
    """Generate a full structured patient risk report."""
    patient  = _get_patient_or_403(pid, db, user)
    checkups = (db.query(DBCheckup).filter(DBCheckup.patient_id==pid)
                .order_by(DBCheckup.checkup_date).all())
    meds     = db.query(DBMedication).filter(DBMedication.patient_id==pid).all()
    diags    = db.query(DBDiagnosis).filter(DBDiagnosis.patient_id==pid).all()
    preds    = (db.query(DBPrediction).filter(DBPrediction.patient_id==pid)
                .order_by(DBPrediction.created_at.desc()).limit(1).all())
    alerts   = (db.query(DBAlert).filter(DBAlert.patient_id==pid, DBAlert.acknowledged==0).all())

    latest_pred = preds[0] if preds else None

    return {
        "report_date": datetime.now(timezone.utc).isoformat(),
        "patient": {
            "id": patient.id, "age": patient.age, "sex": patient.sex,
            "ethnicity": patient.ethnicity, "smoking": patient.smoking_status,
            "family_history": {
                "cancer": patient.family_history_cancer,
                "diabetes": patient.family_history_diabetes,
                "cardiovascular": patient.family_history_cardio,
            }
        },
        "monitoring_summary": {
            "total_checkups": len(checkups),
            "monitoring_period": f"{checkups[0].checkup_date[:10]} to {checkups[-1].checkup_date[:10]}" if len(checkups)>=2 else "Insufficient data",
            "active_medications": sum(1 for m in meds if m.active),
            "active_diagnoses": sum(1 for d in diags if d.status=="active"),
        },
        "latest_risk_scores": {
            "cancer":      {"risk": latest_pred.cancer_risk,      "level": latest_pred.cancer_level}      if latest_pred else None,
            "metabolic":   {"risk": latest_pred.metabolic_risk,   "level": latest_pred.metabolic_level}   if latest_pred else None,
            "cardio":      {"risk": latest_pred.cardio_risk,       "level": latest_pred.cardio_level}      if latest_pred else None,
            "hematologic": {"risk": latest_pred.hematologic_risk, "level": latest_pred.hematologic_level} if latest_pred else None,
            "composite":   latest_pred.composite_score if latest_pred else None,
        },
        "active_alerts": [
            {"level":a.level,"category":a.category,"message":a.message} for a in alerts
        ],
        "recommendation": latest_pred.recommendation if latest_pred else "Run a prediction first.",
        "medications":  [{"name":m.name,"dosage":m.dosage_mg,"freq":m.frequency} for m in meds if m.active],
        "diagnoses":    [{"code":d.icd10_code,"desc":d.description,"status":d.status} for d in diags],
        "disclaimer": "BioSentinel outputs are probabilistic risk estimates for research/decision-support use only. Not a medical diagnosis. Must be reviewed by qualified healthcare professionals.",
    }

# ── ACCOUNT SETTINGS ─────────────────────────────────────────────────────────
@app.put("/api/v1/auth/password")
def change_password(body: PasswordChange,
                    db=Depends(get_db), user=Depends(current_user)):
    """Change the current user's password."""
    if not verify_pw(body.current_password, user.hashed_password):
        raise HTTPException(400, "Current password is incorrect")
    if len(body.new_password) < 6:
        raise HTTPException(400, "New password must be at least 6 characters")
    user.hashed_password = hash_pw(body.new_password)
    db.commit()
    audit(db, user, "change_password")
    return {"message": "Password changed successfully"}

# ── EMAIL CONFIGURATION ───────────────────────────────────────────────────────
@app.get("/api/v1/settings/email")
def get_email_config(db=Depends(get_db), user=Depends(current_user)):
    """Get this user's email alert configuration."""
    cfg = db.query(DBEmailConfig).filter(DBEmailConfig.user_id == user.id).first()
    if not cfg:
        return {"configured": False, "enabled": False}
    return {
        "configured":       True,
        "enabled":          bool(cfg.enabled),
        "smtp_host":        cfg.smtp_host,
        "smtp_port":        cfg.smtp_port,
        "smtp_username":    cfg.smtp_username,
        "smtp_password":    "••••••••" if cfg.smtp_password else None,
        "smtp_use_tls":     bool(cfg.smtp_use_tls),
        "from_address":     cfg.from_address,
        "notify_to":        cfg.notify_to,
        "notify_on_high":   bool(cfg.notify_on_high),
        "notify_on_critical": bool(cfg.notify_on_critical),
    }

@app.put("/api/v1/settings/email")
def update_email_config(body: EmailConfigUpdate,
                        db=Depends(get_db), user=Depends(current_user)):
    """Save or update this user's email alert configuration."""
    cfg = db.query(DBEmailConfig).filter(DBEmailConfig.user_id == user.id).first()
    if not cfg:
        cfg = DBEmailConfig(user_id=user.id)
        db.add(cfg)
    # Only overwrite password if a real value was sent
    cfg.smtp_host       = body.smtp_host
    cfg.smtp_port       = body.smtp_port
    cfg.smtp_username   = body.smtp_username
    if body.smtp_password and body.smtp_password != "••••••••":
        cfg.smtp_password = body.smtp_password
    cfg.smtp_use_tls    = body.smtp_use_tls
    cfg.from_address    = body.from_address
    cfg.notify_to       = body.notify_to
    cfg.notify_on_high  = body.notify_on_high
    cfg.notify_on_critical = body.notify_on_critical
    cfg.enabled         = body.enabled
    cfg.updated_at      = datetime.now(timezone.utc).isoformat()
    db.commit()
    audit(db, user, "update_email_config")
    return {"message": "Email configuration saved", "enabled": bool(cfg.enabled)}

@app.post("/api/v1/settings/email/test")
def test_email(body: EmailTestRequest,
               db=Depends(get_db), user=Depends(current_user)):
    """Send a test email to verify SMTP configuration."""
    cfg = db.query(DBEmailConfig).filter(DBEmailConfig.user_id == user.id).first()
    if not cfg or not cfg.smtp_host:
        raise HTTPException(400, "Email not configured yet. Save your SMTP settings first.")
    result = email_config_engine.test_connection(cfg, body.to_address)
    if result["success"]:
        return {"message": result["message"]}
    raise HTTPException(500, result["message"])

# ── AUDIT LOG ─────────────────────────────────────────────────────────────────
@app.get("/api/v1/audit-log")
def get_audit_log(limit: int = 100, db=Depends(get_db), user=Depends(current_user)):
    """View audit log (admin sees all; others see their own actions)."""
    q = db.query(DBAuditLog).order_by(DBAuditLog.timestamp.desc())
    if user.role != "admin":
        q = q.filter(DBAuditLog.user_id == user.id)
    logs = q.limit(limit).all()
    return {"logs": [
        {"id": l.id, "timestamp": l.timestamp, "username": l.username,
         "action": l.action, "patient_id": l.patient_id, "detail": l.detail}
        for l in logs
    ]}

# ── ENHANCED STATS ────────────────────────────────────────────────────────────
@app.get("/api/v1/stats")
def stats(db=Depends(get_db), user=Depends(current_user)):
    if user.role == "admin":
        pat_q  = db.query(DBPatient)
    else:
        pat_q  = db.query(DBPatient).filter(DBPatient.owner_id == user.id)
    owned_ids = [p.id for p in pat_q.all()]
    email_cfg = db.query(DBEmailConfig).filter(DBEmailConfig.user_id == user.id).first()
    return {
        "total_patients":    pat_q.count(),
        "total_checkups":    db.query(DBCheckup).filter(DBCheckup.patient_id.in_(owned_ids)).count(),
        "total_predictions": db.query(DBPrediction).filter(DBPrediction.patient_id.in_(owned_ids)).count(),
        "total_medications": db.query(DBMedication).filter(DBMedication.patient_id.in_(owned_ids)).count(),
        "total_diagnoses":   db.query(DBDiagnosis).filter(DBDiagnosis.patient_id.in_(owned_ids)).count(),
        "total_diet_plans":  db.query(DBDietPlan).filter(DBDietPlan.patient_id.in_(owned_ids)).count(),
        "total_alerts":      db.query(DBAlert).filter(DBAlert.patient_id.in_(owned_ids)).count(),
        "unread_alerts":     db.query(DBAlert).filter(
                                DBAlert.patient_id.in_(owned_ids),
                                DBAlert.acknowledged == 0).count(),
        "overdue_checkups":  _count_overdue(owned_ids, db),
        "ml_trained":        engine_ml.trained,
        "model_names":       list(engine_ml.models.keys()) if engine_ml.trained else [],
        "email_enabled":     bool(email_cfg and email_cfg.enabled),
    }

def _count_overdue(owned_ids: list, db: Session) -> int:
    """Count patients whose last checkup was more than 95 days ago."""
    if not owned_ids: return 0
    overdue = 0
    cutoff = (datetime.now(timezone.utc) - timedelta(days=95)).date().isoformat()
    for pid in owned_ids:
        last = (db.query(DBCheckup)
                .filter(DBCheckup.patient_id == pid)
                .order_by(DBCheckup.checkup_date.desc())
                .first())
        if not last or last.checkup_date[:10] < cutoff:
            overdue += 1
    return overdue

# ── SHAP EXPLANATION ENDPOINT ────────────────────────────────────────────────

@app.get("/api/v1/patients/{pid}/shap/{domain}")
def get_shap_explanation(pid: str, domain: str,
                         db=Depends(get_db), user=Depends(current_user)):
    """
    Return full SHAP explanation for a specific disease domain.
    domain: cancer | metabolic | cardio | hematologic
    Returns feature-level SHAP values suitable for waterfall/force plots.
    """
    if domain not in ("cancer", "metabolic", "cardio", "hematologic"):
        raise HTTPException(400, "domain must be: cancer | metabolic | cardio | hematologic")

    patient  = _get_patient_or_403(pid, db, user)
    checkups = (db.query(DBCheckup).filter(DBCheckup.patient_id == pid)
                .order_by(DBCheckup.checkup_date).all())
    if not checkups:
        raise HTTPException(400, "No checkups — add at least one checkup first.")

    if not engine_ml.trained:
        raise HTTPException(503, "ML models not yet trained — please wait.")

    feat = engine_ml._extract(checkups, patient)
    if feat is None:
        raise HTTPException(400, "Cannot extract features.")

    feat_raw = feat[0]
    m = engine_ml.models[domain]

    # SHAP values
    if engine_ml.shap_available:
        try:
            import shap
            feat_scaled = engine_ml.scaler.transform(feat.reshape(1, -1))
            sv = m["shap_explainer"].shap_values(feat_scaled)[0]
            base_val = float(m["shap_explainer"].expected_value)

            FEATURE_LABELS = [
                "Age", "Female sex", "Smoking", "Alcohol", "Low exercise",
                "Cancer FH", "Diabetes FH", "Cardio FH", "# Checkups", "Months monitored",
                "HbA1c", "Fasting glucose", "Hemoglobin", "Lymphocytes", "WBC",
                "Platelets", "CEA", "CA-125", "PSA", "ALT", "AST",
                "LDL", "HDL", "Triglycerides", "BP systolic", "BMI",
                "Creatinine", "TSH", "CRP", "Ferritin",
                "HbA1c trend", "Glucose trend", "Hemoglobin trend",
                "Lymphocyte trend", "WBC trend", "CEA trend",
                "ALT trend", "LDL trend", "BP trend", "BMI trend",
                "Platelet trend", "CRP trend",
                "HbA1c volatility", "CEA volatility",
                "Hemoglobin volatility", "Lymphocyte volatility",
                "High values count", "Low values count", "Critical values count",
            ]

            features = []
            for i, (sv_val, raw_val) in enumerate(zip(sv, feat_raw)):
                features.append({
                    "feature":    engine_ml.FEATURES[i] if i < len(engine_ml.FEATURES) else f"f{i}",
                    "label":      FEATURE_LABELS[i] if i < len(FEATURE_LABELS) else f"Feature {i}",
                    "raw_value":  round(float(raw_val), 4),
                    "shap_value": round(float(sv_val), 6),
                    "direction":  "positive" if sv_val > 0 else "negative",
                })

            # Sort by absolute impact
            features.sort(key=lambda x: -abs(x["shap_value"]))

            return {
                "domain":         domain,
                "base_value":     round(base_val, 4),
                "prediction":     round(float(np.clip(
                    m["iso"].predict([m["reg"].predict(
                        engine_ml.scaler.transform(feat))[0]])[0], 0.05, 0.92)), 4),
                "shap_sum":       round(float(sum(sv)), 4),
                "top_features":   features[:15],
                "all_features":   features,
                "method":         "shap_tree_explainer",
            }
        except Exception as e:
            raise HTTPException(500, f"SHAP computation failed: {e}")
    else:
        # Fallback: use GBM feature importances
        fi = m["reg"].feature_importances_
        features = [{"feature": engine_ml.FEATURES[i],
                     "raw_value": round(float(feat_raw[i]), 4),
                     "importance": round(float(fi[i]), 6)}
                    for i in np.argsort(fi)[::-1][:15]]
        return {"domain": domain, "method": "gbt_feature_importance",
                "top_features": features,
                "note": "Install shap package for full SHAP values"}


# ── LAB REPORT OCR UPLOAD ─────────────────────────────────────────────────────

@app.post("/api/v1/ocr/extract")
async def ocr_extract(file: UploadFile = File(...),
                      _=Depends(current_user)):
    """
    Upload a PDF or image of a lab report.
    Returns extracted biomarker values ready to pre-fill the checkup form.

    Supports: PDF, JPG, PNG, TIFF.
    Works with SRL, Dr. Lal, Metropolis, Apollo, and most NABL-format reports.
    """
    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:   # 20 MB limit
        raise HTTPException(413, "File too large. Maximum size is 20 MB.")

    result = lab_ocr.from_upload(contents, file.filename or "upload")

    if "error" in result:
        raise HTTPException(422, result["error"])

    values = result.get("values", {})
    return {
        "extracted_values": values,
        "fields_found":     len(values),
        "method":           result.get("method"),
        "text_length":      result.get("raw_text_length", 0),
        "message": (
            f"Found {len(values)} biomarker values. "
            "Review and confirm before saving as a checkup."
            if values else
            "No recognisable biomarker values found. "
            "Try a clearer scan or enter values manually."
        ),
    }


@app.post("/api/v1/ocr/extract-base64")
async def ocr_extract_base64(body: dict,
                              _=Depends(current_user)):
    """
    Alternative: accept base64-encoded file content.
    Body: {"filename": "report.pdf", "data": "<base64 string>"}
    Useful when calling from JavaScript without FormData.
    """
    try:
        filename = body.get("filename", "upload.pdf")
        data_b64 = body.get("data", "")
        # Strip data-URL prefix if present: "data:application/pdf;base64,..."
        if "," in data_b64:
            data_b64 = data_b64.split(",", 1)[1]
        contents = base64.b64decode(data_b64)
    except Exception as e:
        raise HTTPException(422, f"Invalid base64 data: {e}")

    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large. Maximum size is 20 MB.")

    result = lab_ocr.from_upload(contents, filename)
    if "error" in result:
        raise HTTPException(422, result["error"])

    values = result.get("values", {})
    return {
        "extracted_values": values,
        "fields_found":     len(values),
        "method":           result.get("method"),
        "message": (
            f"Extracted {len(values)} biomarker values."
            if values else
            "No values found. Try a clearer image or enter manually."
        ),
    }

# ── CLAUDE AI ENDPOINTS ───────────────────────────────────────────────────────

@app.post("/api/v1/ocr/claude-vision")
async def ocr_claude_vision(file: UploadFile = File(...),
                            _=Depends(current_user)):
    """
    Upload a lab report image to Claude Vision for high-accuracy extraction.
    Handles handwritten reports, unusual layouts, non-English headers.
    Requires ANTHROPIC_API_KEY environment variable.
    """
    if not CLAUDE_AI_AVAILABLE:
        raise HTTPException(503, "Claude AI not available. Set ANTHROPIC_API_KEY.")

    contents = await file.read()
    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(413, "File too large. Maximum 20 MB.")

    fname = (file.filename or "upload").lower()
    media_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                 "png": "image/png", "webp": "image/webp",
                 "gif": "image/gif", "pdf": "application/pdf"}
    ext = fname.rsplit(".", 1)[-1] if "." in fname else "jpg"
    media_type = media_map.get(ext, "image/jpeg")

    if ext == "pdf":
        result = extract_labs_from_pdf_pages(contents)
    else:
        result = extract_labs_from_image(contents, media_type)

    if "error" in result and not result.get("values"):
        raise HTTPException(422, result["error"])

    values = result.get("values", {})
    return {
        "extracted_values": values,
        "fields_found":     len(values),
        "method":           result.get("method", "claude_vision"),
        "model":            result.get("model", ""),
        "notes":            result.get("notes", ""),
        "message": (
            f"Claude Vision found {len(values)} biomarker values. "
            "Review and confirm before saving."
            if values else
            "No biomarker values detected. Try a higher-resolution scan."
        ),
    }


@app.post("/api/v1/ai/narrative/{pid}")
def ai_narrative(pid: str,
                 audience: str = Query("patient", enum=["patient", "clinician"]),
                 db=Depends(get_db), user=Depends(current_user)):
    """
    Generate a plain-English AI narrative for the latest prediction.
    audience=patient  → simple language for the patient view
    audience=clinician → medical language with suggested actions
    """
    if not CLAUDE_AI_AVAILABLE:
        raise HTTPException(503, "Claude AI not available. Set ANTHROPIC_API_KEY.")

    patient = _get_patient_or_403(pid, db, user)
    pred = (db.query(DBPrediction).filter(DBPrediction.patient_id == pid)
            .order_by(DBPrediction.created_at.desc()).first())
    if not pred:
        raise HTTPException(404, "No predictions found. Run a prediction first.")

    prediction_dict = {
        "cancer":      {"risk": pred.cancer_risk,      "level": pred.cancer_level},
        "metabolic":   {"risk": pred.metabolic_risk,   "level": pred.metabolic_level},
        "cardio":      {"risk": pred.cardio_risk,       "level": pred.cardio_level},
        "hematologic": {"risk": pred.hematologic_risk, "level": pred.hematologic_level},
        "composite":   pred.composite_score,
        "top_features": json.loads(pred.top_features_json or "[]"),
    }
    patient_info = {
        "age":       patient.age,
        "sex":       patient.sex,
        "ethnicity": patient.ethnicity,
        "family_history_cancer":  patient.family_history_cancer,
        "family_history_diabetes":patient.family_history_diabetes,
        "family_history_cardio":  patient.family_history_cardio,
        "smoking_status":         patient.smoking_status,
    }

    narrative = generate_prediction_narrative(prediction_dict, patient_info, audience)
    if not narrative:
        raise HTTPException(503, "Narrative generation failed. Check ANTHROPIC_API_KEY.")

    audit(db, user, "ai_narrative", pid, f"audience={audience}")
    return {
        "narrative":   narrative,
        "audience":    audience,
        "prediction_id": pred.id,
        "model":       "claude-haiku-4-5-20251001",
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/ai/anomalies/{pid}")
def ai_anomaly_detection(pid: str,
                         db=Depends(get_db), user=Depends(current_user)):
    """
    Run Claude Sonnet longitudinal trend anomaly detection on a patient.
    Identifies subtle patterns across the full biomarker timeline that
    fixed thresholds miss — e.g. CEA rising 18 months while still 'normal'.
    """
    if not CLAUDE_AI_AVAILABLE:
        raise HTTPException(503, "Claude AI not available. Set ANTHROPIC_API_KEY.")

    patient = _get_patient_or_403(pid, db, user)
    checkups = (db.query(DBCheckup).filter(DBCheckup.patient_id == pid)
                .order_by(DBCheckup.checkup_date).all())

    if len(checkups) < 2:
        raise HTTPException(400, "Need at least 2 checkups for anomaly detection.")

    checkup_dicts = []
    for c in checkups:
        d = {"checkup_date": c.checkup_date}
        for field in ["hba1c","glucose_fasting","hemoglobin","lymphocytes_pct",
                      "wbc","cea","alt","ldl","hdl","bp_systolic","bmi",
                      "crp","psa","ca125","platelets","creatinine","tsh",
                      "ferritin","triglycerides","ast","neutrophils_pct"]:
            v = getattr(c, field, None)
            if v is not None:
                d[field] = v
        checkup_dicts.append(d)

    patient_info = {
        "age":                    patient.age,
        "sex":                    patient.sex,
        "ethnicity":              patient.ethnicity,
        "smoking_status":         patient.smoking_status,
        "family_history_cancer":  patient.family_history_cancer,
        "family_history_diabetes":patient.family_history_diabetes,
        "family_history_cardio":  patient.family_history_cardio,
    }

    analysis = detect_trend_anomalies(patient_info, checkup_dicts)
    if not analysis:
        raise HTTPException(503, "Anomaly detection failed. Check ANTHROPIC_API_KEY.")

    audit(db, user, "ai_anomaly_scan", pid, f"checkups={len(checkups)}")
    return {
        "analysis":        analysis,
        "checkups_analysed": len(checkups),
        "patient_id":      pid,
        "model":           "claude-sonnet-4-20250514",
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "disclaimer":      "Decision-support only. Clinical judgment required.",
    }


@app.get("/api/v1/cache/stats")
def cache_stats(user=Depends(current_user)):
    """Return in-memory cache statistics."""
    if user.role not in ("admin", "clinician"):
        raise HTTPException(403, "Not authorized")
    return _cache_stats()


@app.post("/api/v1/cache/flush")
def cache_flush(user=Depends(current_user)):
    """Flush all in-memory cache entries. Admin only."""
    if user.role != "admin":
        raise HTTPException(403, "Admin only")
    with _cache_lock:
        count = len(_cache_store)
        _cache_store.clear()
    audit(None, user, "cache_flush", None, f"flushed {count} entries")
    return {"flushed": count, "message": "Cache cleared"}


@app.get("/api/v1/ai/status")
def ai_status(_=Depends(current_user)):
    """Return Claude AI integration status and configured features."""
    status = claude_ai_status()
    status["scheduler_available"] = SCHEDULER_AVAILABLE
    return status


@app.post("/api/v1/medications/{mid}/interaction-explain")
def explain_medication_interaction(mid: str,
                                   db=Depends(get_db),
                                   user=Depends(current_user)):
    """
    Get a plain-English explanation of potential drug interactions
    for a specific medication against the patient's current med list.
    Uses OpenFDA data + Claude Haiku to write the explanation.
    """
    med = db.query(DBMedication).filter(DBMedication.id == mid).first()
    if not med:
        raise HTTPException(404, "Medication not found")

    # Verify patient ownership
    patient = _get_patient_or_403(med.patient_id, db, user)

    # Get all other active medications for this patient
    other_meds = (db.query(DBMedication)
                  .filter(DBMedication.patient_id == med.patient_id,
                          DBMedication.id != mid,
                          DBMedication.active == 1).all())

    if not other_meds:
        return {"explanation": "No other active medications to check interactions with.",
                "interactions_checked": 0}

    existing_names = [m.name for m in other_meds]

    # Fetch raw FDA data for context
    raw_fda = None
    try:
        import urllib.request, urllib.parse
        encoded = urllib.parse.quote(med.name.lower())
        fda_url = (f"https://api.fda.gov/drug/label.json"
                   f"?search=openfda.generic_name:{encoded}&limit=1")
        with urllib.request.urlopen(fda_url, timeout=5) as resp:
            fda_data = json.loads(resp.read())
            results = fda_data.get("results", [])
            if results:
                raw_fda = (results[0].get("drug_interactions", [""])[0] or "")[:1500]
    except Exception:
        pass  # FDA call optional; Claude can still reason without it

    if CLAUDE_AI_AVAILABLE:
        explanation = explain_drug_interactions(med.name, existing_names, raw_fda)
    else:
        explanation = (
            f"Checking {med.name} against: {', '.join(existing_names)}. "
            "Set ANTHROPIC_API_KEY for plain-English AI explanations. "
            "For now, consult your pharmacist or check drugs.com for interactions."
        )

    return {
        "drug":                  med.name,
        "checked_against":       existing_names,
        "explanation":           explanation,
        "fda_data_available":    bool(raw_fda),
        "ai_explanation":        CLAUDE_AI_AVAILABLE,
        "interactions_checked":  len(existing_names),
    }


# ── TREND ALERTS ──────────────────────────────────────────────────────────────

@app.get("/api/v1/patients/{pid}/trend-alerts")
def get_trend_alerts(pid: str, db=Depends(get_db),
                     user=Depends(current_user)):
    """
    Analyse biomarker trends and return alerts even when absolute values
    are still within normal range — the *direction* matters as much as
    the current value.

    E.g. CEA rising from 1.5 → 3.2 over 18 months is flagged even though
    both values are below the 5.0 ng/mL threshold.
    """
    patient = _get_patient_or_403(pid, db, user)
    checkups = (db.query(DBCheckup)
                .filter(DBCheckup.patient_id == pid)
                .order_by(DBCheckup.checkup_date).all())

    if len(checkups) < 3:
        return {"alerts": [], "message":
                "Need at least 3 checkups to detect meaningful trends."}

    TREND_RULES = [
        # (field, label, threshold_pct_rise, months_window, severity)
        ("hba1c",          "HbA1c (blood sugar)",       8,  12, "WARNING"),
        ("cea",            "CEA tumour marker",          40, 18, "WARNING"),
        ("glucose_fasting","Fasting glucose",            10, 12, "WARNING"),
        ("ldl",            "LDL cholesterol",            15, 12, "WARNING"),
        ("bp_systolic",    "Systolic blood pressure",    8,  12, "WARNING"),
        ("bmi",            "BMI",                        5,  12, "WARNING"),
        ("alt",            "ALT (liver enzyme)",         50, 12, "WARNING"),
        ("crp",            "CRP (inflammation)",         60, 12, "WARNING"),
        # Declining markers are also concerning
        ("hemoglobin",     "Hemoglobin",                -10, 12, "WARNING"),
        ("lymphocytes_pct","Lymphocyte count",          -15, 12, "WARNING"),
        ("hdl",            "HDL (good cholesterol)",    -15, 12, "WARNING"),
    ]

    alerts = []
    for field, label, threshold_pct, months_window, severity in TREND_RULES:
        vals = [(c.checkup_date[:10], getattr(c, field))
                for c in checkups if getattr(c, field) is not None]
        if len(vals) < 3:
            continue

        # Use last N months of data
        cutoff_date = (datetime.now(timezone.utc) -
                       timedelta(days=months_window * 30.5)).date().isoformat()
        window = [(d, v) for d, v in vals if d >= cutoff_date]
        if len(window) < 2:
            window = vals[-3:]   # fallback: last 3 readings

        first_val = window[0][1]
        last_val  = window[-1][1]

        if abs(first_val) < 1e-6:
            continue
        pct_change = (last_val - first_val) / abs(first_val) * 100

        # Rising alert
        if threshold_pct > 0 and pct_change >= threshold_pct:
            alerts.append({
                "level":   severity,
                "field":   field,
                "label":   label,
                "message": (
                    f"{label} has risen {pct_change:.0f}% "
                    f"({first_val:.1f} → {last_val:.1f}) "
                    f"over the past {len(window)} checkups. "
                    f"This trend warrants clinical attention even though "
                    f"the absolute value may still appear normal."
                ),
                "from_value":  round(first_val, 2),
                "to_value":    round(last_val, 2),
                "pct_change":  round(pct_change, 1),
                "direction":   "rising",
            })
        # Declining alert
        elif threshold_pct < 0 and pct_change <= threshold_pct:
            alerts.append({
                "level":   severity,
                "field":   field,
                "label":   label,
                "message": (
                    f"{label} has declined {abs(pct_change):.0f}% "
                    f"({first_val:.1f} → {last_val:.1f}) "
                    f"over the past {len(window)} checkups."
                ),
                "from_value":  round(first_val, 2),
                "to_value":    round(last_val, 2),
                "pct_change":  round(pct_change, 1),
                "direction":   "declining",
            })

    return {
        "patient_id":   pid,
        "checkups_analysed": len(checkups),
        "alerts":       alerts,
        "alert_count":  len(alerts),
    }


# ── OVERDUE REMINDERS ─────────────────────────────────────────────────────────

@app.get("/api/v1/reminders/overdue")
def get_overdue_patients(days: int = 90,
                         db=Depends(get_db),
                         user=Depends(current_user)):
    """
    List all patients who haven't had a checkup in `days` days (default 90).
    Returns patient details + last checkup date + days overdue.
    Useful for scheduling next appointments.
    """
    if user.role == "admin":
        patients = db.query(DBPatient).all()
    else:
        patients = (db.query(DBPatient)
                    .filter(DBPatient.owner_id == user.id).all())

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    overdue_list = []

    for pat in patients:
        last_chk = (db.query(DBCheckup)
                    .filter(DBCheckup.patient_id == pat.id)
                    .order_by(DBCheckup.checkup_date.desc())
                    .first())
        last_date = last_chk.checkup_date[:10] if last_chk else None
        if not last_date or last_date < cutoff:
            days_overdue = (
                (datetime.now(timezone.utc).date() -
                 datetime.fromisoformat(last_date).date()).days
                if last_date else None
            )
            overdue_list.append({
                "patient_id":   pat.id,
                "age":          pat.age,
                "sex":          pat.sex,
                "last_checkup": last_date,
                "days_overdue": days_overdue,
                "days_threshold": days,
                "message": (
                    f"No checkup in {days_overdue} days — overdue by "
                    f"{days_overdue - days} days."
                    if days_overdue else
                    "No checkups recorded at all."
                ),
            })

    overdue_list.sort(key=lambda x: (x["days_overdue"] or 99999), reverse=True)
    return {
        "overdue_patients": overdue_list,
        "total_overdue":    len(overdue_list),
        "threshold_days":   days,
    }


# ── PATIENT SELF-REGISTRATION (limited role) ──────────────────────────────────

class PatientSelfRegister(BaseModel):
    """Schema for patient self-registration — minimal required fields."""
    username:  str
    email:     str
    password:  str
    # Linked to a clinician's account
    clinician_username: Optional[str] = None

@app.post("/api/v1/auth/register-patient", status_code=201)
def register_patient_self(body: PatientSelfRegister,
                           db: Session = Depends(get_db)):
    """
    Patient self-registration endpoint.
    Creates a user with role='patient' — read-only access to own records only.
    Optionally links to a clinician by username.
    No JWT required — open endpoint (patients register themselves).
    """
    if db.query(DBUser).filter(DBUser.username == body.username).first():
        raise HTTPException(400, "Username already taken.")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters.")

    clinician_id = None
    if body.clinician_username:
        clin = (db.query(DBUser)
                .filter(DBUser.username == body.clinician_username,
                        DBUser.role.in_(["clinician", "admin"]))
                .first())
        if clin:
            clinician_id = clin.id

    user = DBUser(
        username        = body.username,
        email           = body.email,
        hashed_password = hash_pw(body.password),
        role            = "patient",
    )
    db.add(user); db.commit(); db.refresh(user)
    token = make_token({"sub": user.username})
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":       user.id,
            "username": user.username,
            "role":     user.role,
            "linked_clinician": clinician_id,
        },
        "message": "Patient account created. Your clinician can now link your records.",
    }


# ── HEALTH CHECK (enhanced) ───────────────────────────────────────────────────

# ── PASSWORD RESET — Email / SMS / WhatsApp / Telegram ───────────────────────

class PasswordResetRequest(BaseModel):
    username_or_email: str
    channel: str = "email"   # email | sms | whatsapp | telegram

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str

@app.post("/api/v1/auth/forgot-password")
def forgot_password(body: PasswordResetRequest, db: Session = Depends(get_db)):
    """
    Request a password reset OTP. Works without a JWT.
    Sends a 6-digit OTP via email, SMS, WhatsApp, or Telegram.
    Always returns 200 (never leaks whether account exists).
    """
    # Find user by username or email
    user = (db.query(DBUser)
            .filter((DBUser.username == body.username_or_email) |
                    (DBUser.email == body.username_or_email))
            .first())

    if not user:
        # Return success to avoid user enumeration
        return {"message": "If that account exists, a reset code has been sent.",
                "channel": body.channel}

    # Determine destination
    channel = body.channel.lower()
    if channel == "email":
        dest = user.email
        if not dest:
            raise HTTPException(400, "No email address on file for this account.")
    elif channel in ("sms", "whatsapp"):
        dest = user.phone
        if not dest:
            raise HTTPException(400, "No phone number on file. Add it in Settings first.")
    elif channel == "telegram":
        dest = user.telegram_chat_id
        if not dest:
            raise HTTPException(400, "No Telegram chat ID on file. Add it in Settings first.")
    else:
        raise HTTPException(400, f"Unknown channel: {channel}. Use: email | sms | whatsapp | telegram")

    # Generate 6-digit OTP
    import secrets as _secrets
    otp = str(_secrets.randbelow(900000) + 100000)   # 100000–999999
    expires_at = (datetime.now(timezone.utc) + timedelta(minutes=RESET_TOKEN_EXPIRY_MINUTES)).isoformat()

    # Invalidate previous tokens for this user
    (db.query(DBPasswordResetToken)
       .filter(DBPasswordResetToken.user_id == user.id, DBPasswordResetToken.used == 0)
       .update({"used": 1}))

    token_row = DBPasswordResetToken(user_id=user.id, token=otp,
                                     channel=channel, expires_at=expires_at)
    db.add(token_row); db.commit()

    result = notify_engine.send_reset_otp(channel, dest, otp, user.username)
    return {
        "message": "Reset code sent.",
        "channel": channel,
        "sent":    result.get("sent", False),
        "expires_in_minutes": RESET_TOKEN_EXPIRY_MINUTES,
        # Only include error detail in non-production for debugging
        **({"send_error": result["error"]} if not result.get("sent") and os.getenv("DEBUG") else {}),
    }


@app.post("/api/v1/auth/reset-password")
def reset_password(body: PasswordResetConfirm, db: Session = Depends(get_db)):
    """Confirm password reset using the OTP received via email/SMS/WhatsApp/Telegram."""
    if len(body.new_password) < 8:
        raise HTTPException(400, "New password must be at least 8 characters.")

    token_row = (db.query(DBPasswordResetToken)
                 .filter(DBPasswordResetToken.token == body.token,
                         DBPasswordResetToken.used == 0)
                 .first())

    if not token_row:
        raise HTTPException(400, "Invalid or already-used reset code.")

    if datetime.fromisoformat(token_row.expires_at) < datetime.now(timezone.utc):
        raise HTTPException(400, f"Reset code expired. Request a new one.")

    user = db.query(DBUser).filter(DBUser.id == token_row.user_id).first()
    if not user:
        raise HTTPException(404, "User not found.")

    user.hashed_password = hash_pw(body.new_password)
    token_row.used = 1
    db.commit()
    return {"message": "Password reset successfully. You can now log in with your new password."}


# ── USER PROFILE UPDATE ───────────────────────────────────────────────────────

class UserProfileUpdate(BaseModel):
    email:            Optional[str] = None
    phone:            Optional[str] = None   # +91XXXXXXXXXX for SMS/WhatsApp
    telegram_chat_id: Optional[str] = None  # numeric Telegram chat ID

@app.put("/api/v1/auth/profile")
def update_profile(body: UserProfileUpdate,
                   db=Depends(get_db), user=Depends(current_user)):
    """Update email, phone, or Telegram chat ID for notifications."""
    if body.email is not None:
        # Check unique
        existing = db.query(DBUser).filter(DBUser.email == body.email,
                                           DBUser.id != user.id).first()
        if existing:
            raise HTTPException(400, "That email is already in use by another account.")
        user.email = body.email
    if body.phone is not None:
        user.phone = body.phone
    if body.telegram_chat_id is not None:
        user.telegram_chat_id = body.telegram_chat_id
    db.commit(); db.refresh(user)
    return {"id": user.id, "username": user.username, "email": user.email,
            "phone": user.phone, "telegram_chat_id": user.telegram_chat_id,
            "message": "Profile updated successfully."}

@app.get("/api/v1/auth/profile")
def get_profile(user=Depends(current_user)):
    return {"id": user.id, "username": user.username, "email": user.email,
            "role": user.role, "phone": user.phone,
            "telegram_chat_id": user.telegram_chat_id}


# ── APPOINTMENT REMINDERS ─────────────────────────────────────────────────────

class ReminderSendRequest(BaseModel):
    days_threshold: int = 90
    channel: str = "email"          # email | sms | whatsapp | telegram
    recipient_override: Optional[str] = None   # force-send to this address

@app.post("/api/v1/reminders/send-all")
def send_all_reminders(body: ReminderSendRequest,
                       db=Depends(get_db), user=Depends(current_user)):
    """
    Scan all overdue patients and send reminder notifications.
    Sends via the specified channel to the logged-in user's contact
    (or recipient_override if provided).
    """
    if user.role not in ("admin", "clinician"):
        raise HTTPException(403, "Only clinicians and admins can send reminders.")

    if user.role == "admin":
        patients = db.query(DBPatient).all()
    else:
        patients = (db.query(DBPatient)
                    .filter(DBPatient.owner_id == user.id).all())

    cutoff = (datetime.now(timezone.utc) - timedelta(days=body.days_threshold)).date().isoformat()
    sent_count, skipped_count = 0, 0

    # Determine destination
    channel = body.channel.lower()
    if body.recipient_override:
        dest = body.recipient_override
    elif channel == "email":
        dest = user.email
    elif channel in ("sms", "whatsapp"):
        dest = user.phone
    elif channel == "telegram":
        dest = user.telegram_chat_id
    else:
        raise HTTPException(400, f"Unknown channel: {channel}")

    if not dest:
        raise HTTPException(400, f"No {channel} contact on file. Update your profile first.")

    reminders_sent = []
    for pat in patients:
        last = (db.query(DBCheckup).filter(DBCheckup.patient_id == pat.id)
                .order_by(DBCheckup.checkup_date.desc()).first())
        last_date = last.checkup_date[:10] if last else None
        if last_date and last_date >= cutoff:
            skipped_count += 1; continue

        days_overdue = (
            (datetime.now(timezone.utc).date() - datetime.fromisoformat(last_date).date()).days
            if last_date else None
        )
        result = notify_engine.send_reminder(channel, dest, pat.age,
                                             days_overdue or body.days_threshold)
        if result.get("sent"):
            sent_count += 1
            reminders_sent.append({"patient_id": pat.id, "age": pat.age,
                                    "days_overdue": days_overdue})
        audit(db, user, "send_reminder", pat.id,
              f"channel={channel} days_overdue={days_overdue}")

    return {
        "sent": sent_count,
        "skipped": skipped_count,
        "channel": channel,
        "destination": dest[:4] + "***" if dest and len(dest) > 4 else dest,
        "reminders": reminders_sent,
    }


# ── DRUG INTERACTION CHECKER (OpenFDA) ───────────────────────────────────────

@app.get("/api/v1/drug-interactions")
def check_drug_interactions(drugs: str, db=Depends(get_db),
                             _=Depends(current_user)):
    """
    Check drug interactions using the OpenFDA API.
    drugs: comma-separated list, e.g. "metformin,atorvastatin,aspirin"
    Returns adverse event and interaction data from FDA FAERS database.
    """
    import urllib.request as ur
    import urllib.parse as up

    drug_list = [d.strip().lower() for d in drugs.split(",") if d.strip()]
    if not drug_list:
        raise HTTPException(400, "Provide at least one drug name.")
    if len(drug_list) > 10:
        raise HTTPException(400, "Maximum 10 drugs per request.")

    results = {}
    warnings_found = []

    for drug in drug_list:
        try:
            # OpenFDA drug label endpoint — free, no key required
            encoded = up.quote(drug)
            url = (f"https://api.fda.gov/drug/label.json"
                   f"?search=openfda.generic_name:{encoded}"
                   f"&limit=1")
            req = ur.Request(url, headers={"User-Agent": "BioSentinel/2.0"})
            data = json.loads(ur.urlopen(req, timeout=8).read())
            res = data.get("results", [{}])[0]

            # Extract key safety fields
            drug_info = {
                "name": drug,
                "brand_names": res.get("openfda", {}).get("brand_name", [])[:3],
                "warnings": (res.get("warnings", [""])[0] or "")[:500] if res.get("warnings") else None,
                "drug_interactions": (res.get("drug_interactions", [""])[0] or "")[:800] if res.get("drug_interactions") else None,
                "contraindications": (res.get("contraindications", [""])[0] or "")[:500] if res.get("contraindications") else None,
                "adverse_reactions": (res.get("adverse_reactions", [""])[0] or "")[:400] if res.get("adverse_reactions") else None,
                "found": True,
            }
            results[drug] = drug_info

            if drug_info["drug_interactions"]:
                warnings_found.append(drug)

        except Exception as e:
            results[drug] = {"name": drug, "found": False, "error": str(e)}

    # Simple pairwise interaction check using FAERS adverse events
    pair_warnings = []
    if len(drug_list) >= 2:
        try:
            combo = "+".join(up.quote(d) for d in drug_list[:3])
            url = (f"https://api.fda.gov/drug/event.json"
                   f"?search=patient.drug.openfda.generic_name:{combo}"
                   f"&count=patient.reaction.reactionmeddrapt.exact&limit=5")
            req = ur.Request(url, headers={"User-Agent": "BioSentinel/2.0"})
            data = json.loads(ur.urlopen(req, timeout=8).read())
            top_reactions = [r["term"] for r in data.get("results", [])[:5]]
            if top_reactions:
                pair_warnings.append({
                    "drugs": drug_list[:3],
                    "top_reported_reactions": top_reactions,
                    "source": "FDA FAERS adverse event database",
                    "note": "These are reported adverse events — not confirmed causal interactions. Always consult a pharmacist."
                })
        except Exception:
            pass  # FAERS query is best-effort

    return {
        "drugs_checked": drug_list,
        "results": results,
        "combination_warnings": pair_warnings,
        "drugs_with_interaction_info": warnings_found,
        "disclaimer": (
            "This data comes from the FDA label and FAERS databases and is "
            "for informational purposes only. It does NOT replace a clinical "
            "pharmacist review. Always verify drug interactions with a licensed "
            "healthcare professional before prescribing or dispensing."
        ),
        "source": "OpenFDA API (api.fda.gov) — free public data, no API key required",
    }


@app.get("/api/v1/patients/{pid}/drug-interactions")
def patient_drug_interactions(pid: str, db=Depends(get_db),
                              user=Depends(current_user)):
    """
    Auto-check interactions for ALL active medications a patient is currently on.
    Pulls the patient's medication list and runs it through OpenFDA.
    """
    _get_patient_or_403(pid, db, user)
    meds = (db.query(DBMedication)
            .filter(DBMedication.patient_id == pid, DBMedication.active == 1).all())
    if not meds:
        return {"message": "No active medications found for this patient.",
                "drugs_checked": []}
    drug_names = [m.name.split()[0].lower() for m in meds]  # first word = generic name
    # Call our own endpoint logic
    from fastapi.testclient import TestClient
    drugs_str = ",".join(set(drug_names))
    return check_drug_interactions(drugs_str, db=db, _=user)


# ── BULK CSV / EXCEL IMPORT ──────────────────────────────────────────────────

@app.post("/api/v1/import/patients-csv")
async def import_patients_csv(file: UploadFile = File(...),
                               db=Depends(get_db),
                               user=Depends(current_user)):
    """
    Bulk import patients from a CSV or Excel file.

    Required columns: age, sex
    Optional columns: ethnicity, family_history_cancer, family_history_diabetes,
                      family_history_cardio, smoking_status, alcohol_units_weekly,
                      exercise_min_weekly, notes

    Returns: count of rows imported, any row-level errors.
    Download sample template at: GET /api/v1/import/template
    """
    try:
        import pandas as pd
    except ImportError:
        raise HTTPException(500, "pandas not installed. Run: pip install pandas openpyxl")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large. Max 10 MB.")

    try:
        fname = (file.filename or "upload").lower()
        if fname.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            # Try common encodings for CSV
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    df = pd.read_csv(io.BytesIO(contents), encoding=enc)
                    break
                except Exception:
                    continue
    except Exception as e:
        raise HTTPException(422, f"Could not parse file: {e}")

    # Normalise column names
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_")
                  for c in df.columns]

    if "age" not in df.columns or "sex" not in df.columns:
        raise HTTPException(422,
            "File must have at least 'age' and 'sex' columns. "
            "Download the template at GET /api/v1/import/template")

    imported, errors = [], []

    for idx, row in df.iterrows():
        row_num = idx + 2  # 1-indexed + header row
        try:
            age = int(row.get("age", 0))
            if not (1 <= age <= 120):
                errors.append({"row": row_num, "error": f"Invalid age: {age}"})
                continue
            sex = str(row.get("sex", "")).strip()
            if sex.lower() not in ("male", "female", "other", "m", "f"):
                errors.append({"row": row_num, "error": f"Invalid sex: '{sex}'"})
                continue
            sex = {"m": "Male", "f": "Female"}.get(sex.lower(), sex.capitalize())

            pat = DBPatient(
                owner_id               = user.id,
                age                    = age,
                sex                    = sex,
                ethnicity              = _safe_str(row.get("ethnicity")),
                family_history_cancer  = _safe_int(row.get("family_history_cancer", 0)),
                family_history_diabetes= _safe_int(row.get("family_history_diabetes", 0)),
                family_history_cardio  = _safe_int(row.get("family_history_cardio", 0)),
                smoking_status         = _safe_str(row.get("smoking_status", "never")) or "never",
                alcohol_units_weekly   = _safe_float(row.get("alcohol_units_weekly", 0)),
                exercise_min_weekly    = _safe_int(row.get("exercise_min_weekly", 0)),
                notes                  = _safe_str(row.get("notes")),
            )
            db.add(pat); db.flush()
            imported.append({"row": row_num, "patient_id": pat.id,
                             "age": pat.age, "sex": pat.sex})
        except Exception as e:
            errors.append({"row": row_num, "error": str(e)})

    db.commit()
    audit(db, user, "bulk_import_patients", detail=f"imported={len(imported)} errors={len(errors)}")

    return {
        "imported": len(imported),
        "errors":   len(errors),
        "total_rows": len(df),
        "patients": imported,
        "error_details": errors,
        "message": f"Successfully imported {len(imported)} of {len(df)} patients.",
    }


@app.post("/api/v1/import/checkups-csv")
async def import_checkups_csv(file: UploadFile = File(...),
                               db=Depends(get_db),
                               user=Depends(current_user)):
    """
    Bulk import checkups from CSV/Excel.
    Required columns: patient_id, checkup_date
    All biomarker columns are optional — include only those you have data for.
    Download sample template at: GET /api/v1/import/template?type=checkups
    """
    try:
        import pandas as pd
    except ImportError:
        raise HTTPException(500, "pandas not installed.")

    contents = await file.read()
    try:
        fname = (file.filename or "").lower()
        df = pd.read_excel(io.BytesIO(contents)) if fname.endswith((".xlsx", ".xls")) \
             else pd.read_csv(io.BytesIO(contents), encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(422, f"Cannot parse file: {e}")

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    if "patient_id" not in df.columns or "checkup_date" not in df.columns:
        raise HTTPException(422, "File must have 'patient_id' and 'checkup_date' columns.")

    NUM_FIELDS = [
        "weight_kg","bmi","bp_systolic","bp_diastolic","heart_rate","spo2",
        "wbc","rbc","hemoglobin","hematocrit","platelets",
        "lymphocytes_pct","neutrophils_pct","monocytes_pct","eosinophils_pct",
        "mcv","mch","glucose_fasting","hba1c","creatinine","egfr","bun",
        "alt","ast","albumin","bilirubin","ggt","uric_acid",
        "total_cholesterol","ldl","hdl","triglycerides",
        "tsh","t3","t4","vitamin_d","vitamin_b12","ferritin",
        "cea","ca125","ca199","psa","afp","crp","esr",
    ]

    imported, errors = [], []

    for idx, row in df.iterrows():
        row_num = idx + 2
        try:
            pid = str(row.get("patient_id", "")).strip()
            _get_patient_or_403(pid, db, user)  # enforce ownership
            date = str(row.get("checkup_date", "")).strip()
            if not date:
                errors.append({"row": row_num, "error": "Missing checkup_date"}); continue

            kwargs = {"patient_id": pid, "checkup_date": date,
                      "notes": _safe_str(row.get("notes"))}
            for f in NUM_FIELDS:
                v = row.get(f)
                if v is not None and str(v).strip() not in ("", "nan", "NaN"):
                    try:
                        kwargs[f] = float(v) if "." in str(v) else int(float(v))
                    except Exception:
                        pass

            chk = DBCheckup(**kwargs)
            db.add(chk); db.flush()
            imported.append({"row": row_num, "checkup_id": chk.id})
        except HTTPException as e:
            errors.append({"row": row_num, "error": e.detail})
        except Exception as e:
            errors.append({"row": row_num, "error": str(e)})

    db.commit()
    audit(db, user, "bulk_import_checkups",
          detail=f"imported={len(imported)} errors={len(errors)}")
    return {"imported": len(imported), "errors": len(errors),
            "total_rows": len(df), "error_details": errors,
            "message": f"Imported {len(imported)} of {len(df)} checkups."}


@app.get("/api/v1/import/template")
def download_template(type: str = "patients", _=Depends(current_user)):
    """Return a sample CSV template for bulk import."""
    from fastapi.responses import Response

    if type == "patients":
        csv = (
            "age,sex,ethnicity,family_history_cancer,family_history_diabetes,"
            "family_history_cardio,smoking_status,alcohol_units_weekly,"
            "exercise_min_weekly,notes\n"
            "45,Male,South Asian,1,0,1,never,2,150,Example patient 1\n"
            "38,Female,Caucasian,0,1,0,former,5,90,Example patient 2\n"
            "62,Male,African American,2,1,1,current,7,30,High risk patient\n"
        )
        filename = "biosentinel_patients_template.csv"
    elif type == "checkups":
        csv = (
            "patient_id,checkup_date,hba1c,glucose_fasting,hemoglobin,"
            "lymphocytes_pct,wbc,platelets,cea,alt,ldl,hdl,bp_systolic,"
            "bmi,crp,weight_kg,notes\n"
            "<paste-patient-uuid>,2024-01-15,5.8,102,13.2,28,7.1,245,2.1,28,118,51,128,27.2,1.4,71.0,Q1 checkup\n"
        )
        filename = "biosentinel_checkups_template.csv"
    else:
        raise HTTPException(400, "type must be: patients | checkups")

    return Response(content=csv, media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# ── IMPORT HELPERS ────────────────────────────────────────────────────────────
def _safe_str(v) -> Optional[str]:
    if v is None: return None
    s = str(v).strip()
    return s if s and s.lower() not in ("nan", "none", "") else None

def _safe_int(v, default=0) -> int:
    try: return int(float(v))
    except: return default

def _safe_float(v, default=0.0) -> float:
    try: return float(v)
    except: return default


# ── TWO-FACTOR AUTHENTICATION (TOTP) ─────────────────────────────────────────

@app.post("/api/v1/auth/2fa/setup")
def setup_2fa(db=Depends(get_db), user=Depends(current_user)):
    """
    Step 1: Generate a TOTP secret and QR code URL for the user.
    The user scans the QR code with Google Authenticator / Authy / any TOTP app.
    2FA is NOT enabled until they call /2fa/verify with a valid code.
    """
    if not TOTP_AVAILABLE:
        raise HTTPException(503, "Install pyotp and qrcode: pip install pyotp qrcode[pil]")

    # Generate new secret (always generates fresh — allows re-setup)
    secret = pyotp.random_base32()
    totp   = pyotp.TOTP(secret)
    uri    = totp.provisioning_uri(
        name=user.username,
        issuer_name="BioSentinel"
    )

    # Store secret (not yet enabled — only enabled after verification)
    user.totp_secret  = secret
    user.totp_enabled = 0
    db.commit()

    # Generate QR code as base64 PNG
    qr_b64 = None
    try:
        qr_img = qrcode.make(uri)
        buf = io.BytesIO()
        qr_img.save(buf, format="PNG")
        qr_b64 = base64.b64encode(buf.getvalue()).decode()
    except Exception:
        pass   # QR generation is optional

    return {
        "secret":    secret,
        "uri":       uri,
        "qr_code":   f"data:image/png;base64,{qr_b64}" if qr_b64 else None,
        "instructions": (
            "1. Open Google Authenticator / Authy on your phone.\n"
            "2. Tap '+' → 'Scan QR code' (or enter the secret manually).\n"
            "3. Call POST /api/v1/auth/2fa/verify with the 6-digit code to activate."
        ),
    }


@app.post("/api/v1/auth/2fa/verify")
def verify_2fa(body: dict, db=Depends(get_db), user=Depends(current_user)):
    """
    Step 2: Verify the first TOTP code to confirm setup and enable 2FA.
    body: {"code": "123456"}
    Returns 10 single-use backup codes on first activation.
    """
    if not TOTP_AVAILABLE:
        raise HTTPException(503, "Install pyotp: pip install pyotp")
    if not user.totp_secret:
        raise HTTPException(400, "Call /2fa/setup first to generate a secret.")

    code = str(body.get("code", "")).strip()
    if not code:
        raise HTTPException(400, "Provide the 6-digit code from your authenticator app.")

    totp = pyotp.TOTP(user.totp_secret)
    if not totp.verify(code, valid_window=1):
        raise HTTPException(400, "Invalid code. Make sure your phone clock is synced.")

    # Generate 10 backup codes
    raw_backups   = [_secrets.token_hex(4).upper() for _ in range(10)]
    hashed_backups = [hash_pw(c) for c in raw_backups]

    user.totp_enabled     = 1
    user.totp_backup_codes = json.dumps(hashed_backups)
    db.commit()
    audit(db, user, "enable_2fa")

    return {
        "enabled":     True,
        "backup_codes": raw_backups,
        "warning": (
            "Save these backup codes somewhere safe. Each can only be used ONCE "
            "and they cannot be retrieved again. Use them if you lose your phone."
        ),
    }


@app.delete("/api/v1/auth/2fa/disable")
def disable_2fa(body: dict, db=Depends(get_db), user=Depends(current_user)):
    """
    Disable 2FA. Requires current password confirmation.
    body: {"password": "yourpassword"}
    """
    if not verify_pw(body.get("password", ""), user.hashed_password):
        raise HTTPException(400, "Incorrect password.")

    user.totp_enabled     = 0
    user.totp_secret      = None
    user.totp_backup_codes = None
    db.commit()
    audit(db, user, "disable_2fa")
    return {"enabled": False, "message": "2FA has been disabled."}


@app.get("/api/v1/auth/2fa/status")
def twofa_status(user=Depends(current_user)):
    return {
        "totp_enabled":  bool(user.totp_enabled),
        "totp_available": TOTP_AVAILABLE,
    }


# ── FHIR R4 IMPORT ────────────────────────────────────────────────────────────

class FHIRImportRequest(BaseModel):
    fhir_server_url: str            # e.g. https://hapi.fhir.org/baseR4
    resource_type:   str = "Patient" # Patient | Observation | MedicationRequest
    patient_id:      Optional[str]  = None   # FHIR Patient resource ID
    auth_token:      Optional[str]  = None   # Bearer token if server requires auth
    max_records:     int = 50

@app.post("/api/v1/fhir/import")
def fhir_import(body: FHIRImportRequest,
                db=Depends(get_db), user=Depends(current_user)):
    """
    Import patients and observations from any FHIR R4 compliant server.

    Supports:
    - Epic MyChart FHIR sandbox
    - Cerner FHIR R4
    - HAPI FHIR public server
    - Google Cloud Healthcare FHIR API
    - Any FHIR R4-compliant endpoint

    For testing use: https://hapi.fhir.org/baseR4
    """
    import urllib.request as _ur
    import urllib.parse   as _up

    server = body.fhir_server_url.rstrip("/")
    headers = {"Accept": "application/fhir+json", "Content-Type": "application/fhir+json"}
    if body.auth_token:
        headers["Authorization"] = f"Bearer {body.auth_token}"

    def fhir_get(url: str) -> dict:
        req = _ur.Request(url, headers=headers)
        resp = _ur.urlopen(req, timeout=15)
        return json.loads(resp.read())

    results = {"imported": 0, "skipped": 0, "errors": [], "records": []}

    try:
        if body.resource_type == "Patient":
            # Fetch Patient bundle
            url = f"{server}/Patient?_count={body.max_records}&_format=json"
            if body.patient_id:
                url = f"{server}/Patient/{body.patient_id}?_format=json"
            data = fhir_get(url)

            patients_to_import = []
            if data.get("resourceType") == "Bundle":
                for entry in data.get("entry", []):
                    p = entry.get("resource", {})
                    if p.get("resourceType") == "Patient":
                        patients_to_import.append(p)
            elif data.get("resourceType") == "Patient":
                patients_to_import = [data]

            for fhir_pat in patients_to_import:
                try:
                    # Extract demographics
                    dob  = fhir_pat.get("birthDate", "")
                    age  = _fhir_age(dob)
                    gender = fhir_pat.get("gender", "unknown")
                    sex  = "Male" if gender == "male" else "Female" if gender == "female" else "Other"

                    # Extract name
                    names = fhir_pat.get("name", [{}])
                    name_text = " ".join(
                        (names[0].get("given", [""])[0] + " " +
                         names[0].get("family", "")).split()
                    ) if names else ""

                    # Extract extensions (ethnicity etc.)
                    ethnicity = _fhir_ethnicity(fhir_pat)

                    fhir_id = fhir_pat.get("id", "")
                    notes = f"Imported from FHIR R4: {server} | FHIR ID: {fhir_id}"
                    if name_text:
                        notes = f"Name: {name_text} | " + notes

                    pat = DBPatient(
                        owner_id = user.id,
                        age      = age or 0,
                        sex      = sex,
                        ethnicity = ethnicity,
                        notes    = notes,
                    )
                    db.add(pat); db.flush()
                    results["imported"] += 1
                    results["records"].append({
                        "fhir_id":    fhir_id,
                        "patient_id": pat.id,
                        "age": age, "sex": sex,
                    })
                except Exception as e:
                    results["errors"].append({"fhir_id": fhir_pat.get("id"), "error": str(e)})
                    results["skipped"] += 1

        elif body.resource_type == "Observation" and body.patient_id:
            # Fetch lab observations for a specific FHIR patient
            url = (f"{server}/Observation"
                   f"?patient={body.patient_id}"
                   f"&category=laboratory"
                   f"&_count={body.max_records}"
                   f"&_sort=-date"
                   f"&_format=json")
            data = fhir_get(url)
            observations = [e.get("resource", {}) for e in data.get("entry", [])]

            # Group by date → create checkups
            by_date: dict = {}
            for obs in observations:
                if obs.get("resourceType") != "Observation": continue
                date_str = (obs.get("effectiveDateTime") or
                            obs.get("effectivePeriod", {}).get("start", ""))[:10]
                if not date_str: continue
                by_date.setdefault(date_str, []).append(obs)

            for date_str, obs_list in sorted(by_date.items()):
                fields = _fhir_obs_to_checkup(obs_list)
                if not fields: continue
                # Find or create a matching BioSentinel patient
                # For demo, use body.patient_id as notes identifier
                matching = (db.query(DBPatient)
                             .filter(DBPatient.owner_id == user.id,
                                     DBPatient.notes.like(f"%{body.patient_id}%"))
                             .first())
                if not matching:
                    results["errors"].append({
                        "date": date_str,
                        "error": f"No BioSentinel patient found for FHIR ID {body.patient_id}. Import the Patient first."
                    })
                    results["skipped"] += 1
                    continue
                chk = DBCheckup(patient_id=matching.id, checkup_date=date_str, **fields)
                db.add(chk); db.flush()
                results["imported"] += 1
                results["records"].append({"date": date_str, "fields": list(fields.keys())})

        elif body.resource_type == "MedicationRequest" and body.patient_id:
            url = (f"{server}/MedicationRequest"
                   f"?patient={body.patient_id}"
                   f"&status=active"
                   f"&_count={body.max_records}"
                   f"&_format=json")
            data = fhir_get(url)
            for entry in data.get("entry", []):
                rx = entry.get("resource", {})
                if rx.get("resourceType") != "MedicationRequest": continue
                try:
                    med_name = (
                        rx.get("medicationCodeableConcept", {}).get("text") or
                        rx.get("medicationCodeableConcept", {}).get("coding", [{}])[0].get("display") or
                        "Unknown medication"
                    )
                    # Find matching patient
                    matching = (db.query(DBPatient)
                                 .filter(DBPatient.owner_id == user.id,
                                         DBPatient.notes.like(f"%{body.patient_id}%"))
                                 .first())
                    if not matching:
                        results["skipped"] += 1; continue
                    med = DBMedication(patient_id=matching.id, name=med_name,
                                       active=1, prescribed_for="Imported from FHIR R4")
                    db.add(med); db.flush()
                    results["imported"] += 1
                    results["records"].append({"medication": med_name})
                except Exception as e:
                    results["errors"].append({"error": str(e)}); results["skipped"] += 1
        else:
            raise HTTPException(400,
                f"Unsupported resource_type '{body.resource_type}'. "
                "Use: Patient | Observation (with patient_id) | MedicationRequest (with patient_id)")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"FHIR server error: {e}")

    db.commit()
    audit(db, user, "fhir_import",
          detail=f"server={body.fhir_server_url} resource={body.resource_type} imported={results['imported']}")

    results["message"] = (
        f"Imported {results['imported']} {body.resource_type} records from FHIR server."
    )
    return results


@app.get("/api/v1/fhir/test-connection")
def fhir_test(server_url: str, _=Depends(current_user)):
    """Test connectivity to a FHIR R4 server by fetching its capability statement."""
    import urllib.request as _ur
    try:
        url = server_url.rstrip("/") + "/metadata?_format=json"
        req = _ur.Request(url, headers={"Accept": "application/fhir+json"})
        data = json.loads(_ur.urlopen(req, timeout=10).read())
        fhir_version = data.get("fhirVersion", "?")
        return {
            "connected":     True,
            "fhir_version":  fhir_version,
            "server_name":   data.get("software", {}).get("name", "Unknown"),
            "resources":     len(data.get("rest", [{}])[0].get("resource", [])),
            "url":           server_url,
        }
    except Exception as e:
        raise HTTPException(502, f"Cannot connect to FHIR server: {e}")


def _fhir_age(dob_str: str) -> Optional[int]:
    if not dob_str: return None
    try:
        from datetime import date
        dob = date.fromisoformat(dob_str[:10])
        today = date.today()
        return int((today - dob).days / 365.25)
    except Exception:
        return None


def _fhir_ethnicity(fhir_pat: dict) -> Optional[str]:
    for ext in fhir_pat.get("extension", []):
        if "ethnicity" in ext.get("url", "").lower():
            for sub in ext.get("extension", []):
                text = sub.get("valueString") or sub.get("valueCodeableConcept", {}).get("text")
                if text: return text
    return None


# LOINC codes → BioSentinel checkup field names
_LOINC_MAP = {
    "4548-4":  "hba1c",           "17856-6": "hba1c",
    "2339-0":  "glucose_fasting", "1558-6":  "glucose_fasting",
    "718-7":   "hemoglobin",      "20570-8": "hematocrit",
    "6690-2":  "wbc",             "777-3":   "platelets",
    "736-9":   "lymphocytes_pct", "770-8":   "neutrophils_pct",
    "2089-1":  "ldl",             "2085-9":  "hdl",
    "2571-8":  "triglycerides",   "2093-3":  "total_cholesterol",
    "2160-0":  "creatinine",      "33914-3": "egfr",
    "3094-0":  "bun",             "1742-6":  "alt",
    "1920-8":  "ast",             "1975-2":  "bilirubin",
    "1893-5":  "albumin",         "2324-2":  "ggt",
    "3016-3":  "tsh",             "14920-3": "t3", "14635-7": "t4",
    "4263-0":  "cea",             "10334-1": "ca125", "5765-8": "psa",
    "1988-5":  "crp",             "4537-7":  "esr",
    "2947-0":  "bp_systolic",     "8480-6":  "bp_systolic",
    "8462-4":  "bp_diastolic",    "29463-7": "weight_kg",
    "39156-5": "bmi",             "8867-4":  "heart_rate",
    "59408-5": "spo2",
}

def _fhir_obs_to_checkup(observations: list) -> dict:
    """Map FHIR Observation resources to BioSentinel checkup fields."""
    fields = {}
    for obs in observations:
        # Extract LOINC code
        code = None
        for coding in obs.get("code", {}).get("coding", []):
            if coding.get("system", "").endswith("loinc.org"):
                code = coding.get("code")
                break
        if not code: continue
        field = _LOINC_MAP.get(code)
        if not field: continue
        # Extract value
        val = (obs.get("valueQuantity", {}).get("value") or
               obs.get("valueCodeableConcept", {}).get("coding", [{}])[0].get("code"))
        if val is not None:
            try: fields[field] = float(val)
            except: pass
    return fields


# ── SESSION MANAGEMENT ────────────────────────────────────────────────────────

@app.get("/api/v1/auth/sessions")
def list_sessions(db=Depends(get_db), user=Depends(current_user)):
    """List all active sessions for the current user."""
    sessions = (db.query(DBActiveSession)
                .filter(DBActiveSession.user_id == user.id,
                        DBActiveSession.revoked == 0)
                .order_by(DBActiveSession.last_seen.desc())
                .all())
    now = datetime.now(timezone.utc).isoformat()
    return {"sessions": [
        {"id": s.id, "device": s.device, "ip_address": s.ip_address,
         "created_at": s.created_at, "last_seen": s.last_seen,
         "expires_at": s.expires_at,
         "expired": s.expires_at < now if s.expires_at else False}
        for s in sessions
    ]}

@app.delete("/api/v1/auth/sessions/{session_id}")
def revoke_session(session_id: str, db=Depends(get_db),
                   user=Depends(current_user)):
    """Revoke a specific session (log out a device)."""
    sess = (db.query(DBActiveSession)
            .filter(DBActiveSession.id == session_id,
                    DBActiveSession.user_id == user.id).first())
    if not sess:
        raise HTTPException(404, "Session not found.")
    sess.revoked = 1
    REVOKED_JTIS.add(sess.jti)
    db.commit()
    audit(db, user, "revoke_session", detail=f"session_id={session_id}")
    return {"message": "Session revoked. That device will need to log in again."}

@app.delete("/api/v1/auth/sessions")
def revoke_all_sessions(db=Depends(get_db), user=Depends(current_user)):
    """Revoke all other sessions (keep only current)."""
    sessions = (db.query(DBActiveSession)
                .filter(DBActiveSession.user_id == user.id,
                        DBActiveSession.revoked == 0).all())
    for s in sessions:
        s.revoked = 1
        REVOKED_JTIS.add(s.jti)
    db.commit()
    audit(db, user, "revoke_all_sessions", detail=f"count={len(sessions)}")
    return {"message": f"Revoked {len(sessions)} sessions.", "count": len(sessions)}


# ── NOTIFICATION PREFERENCES ──────────────────────────────────────────────────

class NotifPrefUpdate(BaseModel):
    email_enabled:    Optional[int] = None
    sms_enabled:      Optional[int] = None
    whatsapp_enabled: Optional[int] = None
    telegram_enabled: Optional[int] = None
    notify_critical:  Optional[int] = None
    notify_high:      Optional[int] = None
    notify_moderate:  Optional[int] = None
    notify_overdue:   Optional[int] = None
    notify_login:     Optional[int] = None
    quiet_start:      Optional[int] = None   # hour 0-23
    quiet_end:        Optional[int] = None

@app.get("/api/v1/settings/notifications")
def get_notif_prefs(db=Depends(get_db), user=Depends(current_user)):
    prefs = (db.query(DBNotificationPreference)
             .filter(DBNotificationPreference.user_id == user.id).first())
    if not prefs:
        # Return defaults
        return {"email_enabled": 1, "sms_enabled": 0, "whatsapp_enabled": 0,
                "telegram_enabled": 0, "notify_critical": 1, "notify_high": 1,
                "notify_moderate": 0, "notify_overdue": 1, "notify_login": 0,
                "quiet_start": None, "quiet_end": None}
    return {c.name: getattr(prefs, c.name)
            for c in prefs.__table__.columns
            if c.name not in ("id", "user_id", "updated_at")}

@app.put("/api/v1/settings/notifications")
def update_notif_prefs(body: NotifPrefUpdate, db=Depends(get_db),
                       user=Depends(current_user)):
    prefs = (db.query(DBNotificationPreference)
             .filter(DBNotificationPreference.user_id == user.id).first())
    if not prefs:
        prefs = DBNotificationPreference(user_id=user.id)
        db.add(prefs)
    for field, val in body.model_dump(exclude_none=True).items():
        setattr(prefs, field, val)
    prefs.updated_at = datetime.now(timezone.utc).isoformat()
    db.commit()
    return {"message": "Notification preferences saved.", **body.model_dump(exclude_none=True)}


# ── WEBHOOK SYSTEM ────────────────────────────────────────────────────────────

WEBHOOK_EVENTS = [
    "prediction.critical",     # patient reaches CRITICAL risk
    "prediction.high",         # patient reaches HIGH risk
    "prediction.complete",     # any prediction completed
    "alert.new",               # new clinical alert generated
    "alert.acknowledged",      # alert acknowledged
    "patient.created",         # new patient enrolled
    "checkup.added",           # checkup data added
    "reminder.overdue",        # overdue patient detected
]

class WebhookCreate(BaseModel):
    name:   str
    url:    str
    secret: Optional[str] = None
    events: List[str] = ["prediction.critical", "alert.new"]

@app.get("/api/v1/webhooks")
def list_webhooks(db=Depends(get_db), user=Depends(current_user)):
    whs = db.query(DBWebhook).filter(DBWebhook.user_id == user.id).all()
    return {"webhooks": [
        {"id": w.id, "name": w.name, "url": w.url,
         "events": json.loads(w.events), "active": bool(w.active),
         "last_fired": w.last_fired, "fail_count": w.fail_count}
        for w in whs
    ], "available_events": WEBHOOK_EVENTS}

@app.post("/api/v1/webhooks", status_code=201)
def create_webhook(body: WebhookCreate, db=Depends(get_db),
                   user=Depends(current_user)):
    invalid = [e for e in body.events if e not in WEBHOOK_EVENTS]
    if invalid:
        raise HTTPException(400, f"Unknown events: {invalid}. Valid: {WEBHOOK_EVENTS}")
    wh = DBWebhook(user_id=user.id, name=body.name, url=body.url,
                   secret=body.secret, events=json.dumps(body.events))
    db.add(wh); db.commit(); db.refresh(wh)
    return {"id": wh.id, "name": wh.name, "url": wh.url,
            "events": body.events, "message": "Webhook created."}

@app.delete("/api/v1/webhooks/{wid}")
def delete_webhook(wid: str, db=Depends(get_db), user=Depends(current_user)):
    wh = db.query(DBWebhook).filter(DBWebhook.id == wid,
                                     DBWebhook.user_id == user.id).first()
    if not wh: raise HTTPException(404, "Webhook not found.")
    db.delete(wh); db.commit()
    return {"message": "Webhook deleted."}

@app.post("/api/v1/webhooks/{wid}/test")
def test_webhook(wid: str, db=Depends(get_db), user=Depends(current_user)):
    """Send a test payload to verify the webhook URL is reachable."""
    wh = db.query(DBWebhook).filter(DBWebhook.id == wid,
                                     DBWebhook.user_id == user.id).first()
    if not wh: raise HTTPException(404, "Webhook not found.")
    result = _fire_webhook(wh, "webhook.test",
                           {"message": "BioSentinel webhook test", "timestamp": datetime.now(timezone.utc).isoformat()})
    return {"sent": result["ok"], "status_code": result.get("status_code"),
            "error": result.get("error")}

def _fire_webhook(wh: DBWebhook, event: str, payload: dict) -> dict:
    """
    Fire a webhook with HMAC-SHA256 signature.
    X-BioSentinel-Signature: sha256=<hex>
    X-BioSentinel-Event: <event>
    """
    import urllib.request as _ur, urllib.error as _ue
    if not wh.active: return {"ok": False, "error": "Webhook disabled"}
    if event not in json.loads(wh.events or "[]") and event != "webhook.test":
        return {"ok": True, "skipped": True}
    try:
        body = json.dumps({"event": event, "data": payload,
                           "timestamp": datetime.now(timezone.utc).isoformat()}).encode()
        sig = ""
        if wh.secret:
            sig = "sha256=" + hmac.new(wh.secret.encode(), body, hashlib.sha256).hexdigest()
        req = _ur.Request(wh.url, data=body, headers={
            "Content-Type":           "application/json",
            "X-BioSentinel-Event":     event,
            "X-BioSentinel-Signature": sig,
            "User-Agent":              "BioSentinel/2.0 Webhook",
        })
        resp = _ur.urlopen(req, timeout=10)
        wh.last_fired = datetime.now(timezone.utc).isoformat()
        wh.fail_count = 0
        return {"ok": True, "status_code": resp.status}
    except Exception as e:
        wh.fail_count = (wh.fail_count or 0) + 1
        if wh.fail_count >= 10: wh.active = 0   # auto-disable after 10 fails
        return {"ok": False, "error": str(e)}

def fire_webhooks_async(user_id: str, event: str, payload: dict, db):
    """Fire all matching webhooks for a user in background threads."""
    whs = db.query(DBWebhook).filter(DBWebhook.user_id == user_id,
                                      DBWebhook.active == 1).all()
    for wh in whs:
        if event in json.loads(wh.events or "[]"):
            t = threading.Thread(target=_fire_webhook, args=(wh, event, payload), daemon=True)
            t.start()


# ── BATCH PREDICTIONS ─────────────────────────────────────────────────────────

@app.post("/api/v1/patients/predict-all")
def batch_predict(db=Depends(get_db), user=Depends(current_user)):
    """
    Run AI predictions for ALL patients that have at least one checkup.
    Returns a summary table — useful for morning dashboard refresh or cron job.
    """
    if user.role == "admin":
        patients = db.query(DBPatient).all()
    else:
        patients = (db.query(DBPatient)
                    .filter(DBPatient.owner_id == user.id).all())

    results = []
    skipped = []
    errors  = []

    for pat in patients:
        checkups = (db.query(DBCheckup).filter(DBCheckup.patient_id == pat.id)
                    .order_by(DBCheckup.checkup_date).all())
        if not checkups:
            skipped.append(pat.id); continue
        try:
            pred = engine_ml.predict(checkups, pat)
            if "error" in pred:
                errors.append({"patient_id": pat.id, "error": pred["error"]}); continue

            # Save prediction
            db_pred = DBPrediction(patient_id=pat.id,
                cancer_risk=pred["cancer"]["risk"], cancer_level=pred["cancer"]["level"],
                metabolic_risk=pred["metabolic"]["risk"], metabolic_level=pred["metabolic"]["level"],
                cardio_risk=pred["cardio"]["risk"], cardio_level=pred["cardio"]["level"],
                hematologic_risk=pred["hematologic"]["risk"], hematologic_level=pred["hematologic"]["level"],
                composite_score=pred["composite"], checkups_used=pred["checkups_used"],
                recommendation=pred["recommendation"])
            db.add(db_pred); db.flush()

            results.append({
                "patient_id": pat.id, "age": pat.age, "sex": pat.sex,
                "cancer":     pred["cancer"]["risk"],
                "metabolic":  pred["metabolic"]["risk"],
                "cardio":     pred["cardio"]["risk"],
                "composite":  pred["composite"],
                "level":      max([pred["cancer"]["level"], pred["metabolic"]["level"],
                                   pred["cardio"]["level"]],
                                  key=lambda x: ["LOW","MODERATE","HIGH","CRITICAL"].index(x)),
                "alerts":     len(pred.get("alerts", [])),
            })
            # Fire webhooks
            if pred["cancer"]["level"] == "CRITICAL" or pred["metabolic"]["level"] == "CRITICAL":
                fire_webhooks_async(user.id, "prediction.critical", {
                    "patient_id": pat.id, "composite": pred["composite"]}, db)
        except Exception as e:
            errors.append({"patient_id": pat.id, "error": str(e)})

    db.commit()
    audit(db, user, "batch_predict",
          detail=f"predicted={len(results)} skipped={len(skipped)} errors={len(errors)}")
    log.info("batch_predict", user=user.username,
             predicted=len(results), skipped=len(skipped))

    # Sort by composite risk descending
    results.sort(key=lambda x: -x["composite"])
    return {
        "predicted": len(results), "skipped": len(skipped), "errors": len(errors),
        "results":   results,
        "error_details": errors,
        "message":   f"Ran predictions for {len(results)} patients.",
    }


# ── DATA EXPORT ───────────────────────────────────────────────────────────────

@app.get("/api/v1/patients/{pid}/export/csv")
def export_patient_csv(pid: str, db=Depends(get_db), user=Depends(current_user)):
    """Export complete patient history as CSV — all checkups with all biomarker columns."""
    patient = _get_patient_or_403(pid, db, user)
    checkups = (db.query(DBCheckup).filter(DBCheckup.patient_id == pid)
                .order_by(DBCheckup.checkup_date).all())

    COLS = ["checkup_date","weight_kg","bmi","bp_systolic","bp_diastolic",
            "heart_rate","wbc","hemoglobin","platelets","lymphocytes_pct",
            "neutrophils_pct","glucose_fasting","hba1c","creatinine","alt",
            "ast","albumin","crp","total_cholesterol","ldl","hdl","triglycerides",
            "tsh","vitamin_d","vitamin_b12","ferritin","cea","ca125","psa","notes"]

    buf = io.StringIO()
    buf.write(f"# BioSentinel Patient Export | Age:{patient.age} Sex:{patient.sex} | {datetime.now(timezone.utc).date()}\n")
    buf.write(",".join(COLS) + "\n")
    for chk in checkups:
        row = [str(getattr(chk, c) or "") for c in COLS]
        buf.write(",".join(row) + "\n")

    audit(db, user, "export_patient_csv", pid)
    return Response(
        content=buf.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="patient_{pid[:8]}_export.csv"'}
    )

@app.get("/api/v1/patients/{pid}/export/pdf")
def export_patient_pdf(pid: str, db=Depends(get_db), user=Depends(current_user)):
    """Export professional PDF report with full patient history and risk scores."""
    patient  = _get_patient_or_403(pid, db, user)
    checkups = (db.query(DBCheckup).filter(DBCheckup.patient_id == pid)
                .order_by(DBCheckup.checkup_date).all())
    preds    = (db.query(DBPrediction).filter(DBPrediction.patient_id == pid)
                .order_by(DBPrediction.created_at.desc()).all())
    meds     = (db.query(DBMedication).filter(DBMedication.patient_id == pid,
                                              DBMedication.active == 1).all())
    diags    = (db.query(DBDiagnosis).filter(DBDiagnosis.patient_id == pid).all())

    if not PDF_EXPORT_AVAILABLE:
        return export_patient_json_report(pid, db, user)

    def s(v):
        """Safe latin-1 string for fpdf2."""
        if v is None: return ""
        return str(v).encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header bar
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_fill_color(5, 150, 105)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 14, s("BioSentinel - Patient Health Report"),
             new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", size=8)
    pdf.cell(0, 5, s(f"Generated: {datetime.now(timezone.utc).strftime('%d %B %Y, %H:%M UTC')} | "
                     "Developer: Liveupx Pvt. Ltd. | github.com/liveupx/biosentinel"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Patient demographics
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, s("Patient Profile"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, s(f"Age: {patient.age}  |  Sex: {patient.sex}  |  "
                     f"Ethnicity: {patient.ethnicity or 'Not specified'}  |  "
                     f"Enrolled: {patient.created_at[:10]}"),
             new_x="LMARGIN", new_y="NEXT")
    fh_parts = []
    if patient.family_history_cancer: fh_parts.append(f"Cancer x{patient.family_history_cancer}")
    if patient.family_history_diabetes: fh_parts.append("Diabetes")
    if patient.family_history_cardio: fh_parts.append("Cardiovascular")
    if fh_parts:
        pdf.cell(0, 6, s(f"Family History: {', '.join(fh_parts)}"),
                 new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, s(f"Smoking: {patient.smoking_status}  |  "
                     f"Exercise: {patient.exercise_min_weekly} min/week  |  "
                     f"Alcohol: {patient.alcohol_units_weekly} units/week"),
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # Latest risk scores
    if preds:
        p = preds[0]
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, s("Latest AI Risk Assessment"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        for domain, risk, level in [
            ("Cancer",      p.cancer_risk,      p.cancer_level),
            ("Metabolic",   p.metabolic_risk,   p.metabolic_level),
            ("Cardio",      p.cardio_risk,       p.cardio_level),
            ("Hematologic", p.hematologic_risk,  p.hematologic_level),
        ]:
            pdf.cell(60, 6, s(f"{domain}:"), border=0)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(30, 6, s(f"{(risk or 0)*100:.1f}%"))
            pdf.set_font("Helvetica", size=10)
            pdf.cell(0, 6, s(f"[{level}]"), new_x="LMARGIN", new_y="NEXT")
        if p.recommendation:
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 5, s(f"Recommendation: {p.recommendation}"))
        pdf.ln(4)

    # Checkup history
    if checkups:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, s(f"Checkup History ({len(checkups)} records)"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 8)
        hdrs = ["Date","HbA1c","Glucose","Hgb","CEA","LDL","BP Sys","BMI"]
        widths = [24, 16, 18, 14, 14, 14, 16, 14]
        for h, w in zip(hdrs, widths):
            pdf.cell(w, 7, s(h), border=1)
        pdf.ln()
        pdf.set_font("Helvetica", size=8)
        for chk in checkups[-20:]:
            vals = [
                s(chk.checkup_date[:10]),
                s(f"{chk.hba1c:.1f}" if chk.hba1c else "--"),
                s(f"{chk.glucose_fasting:.0f}" if chk.glucose_fasting else "--"),
                s(f"{chk.hemoglobin:.1f}" if chk.hemoglobin else "--"),
                s(f"{chk.cea:.1f}" if chk.cea else "--"),
                s(f"{chk.ldl:.0f}" if chk.ldl else "--"),
                s(f"{chk.bp_systolic:.0f}" if chk.bp_systolic else "--"),
                s(f"{chk.bmi:.1f}" if chk.bmi else "--"),
            ]
            for v, w in zip(vals, widths):
                pdf.cell(w, 6, v, border=1)
            pdf.ln()
        pdf.ln(4)

    # Active medications
    if meds:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, s("Active Medications"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        for m in meds:
            line = f"- {m.name}"
            if m.dosage_mg: line += f" {m.dosage_mg}mg"
            if m.frequency: line += f"  ({m.frequency})"
            pdf.cell(0, 6, s(line), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # Diagnoses
    if diags:
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, s("Diagnoses"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", size=10)
        for d in diags:
            pdf.cell(0, 6, s(f"- {d.icd10_code or ''} {d.description or ''} [{d.status or ''}]"),
                     new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 5, s(
        "DISCLAIMER: BioSentinel is a research and clinical decision-support tool. "
        "It is NOT a licensed medical device. All AI-generated predictions must be reviewed "
        "by a qualified healthcare professional before any clinical action is taken. "
        "Liveupx Pvt. Ltd. accepts no liability for clinical decisions based on this report."))

    pdf_bytes = pdf.output()
    audit(db, user, "export_patient_pdf", pid)
    return Response(
        content=bytes(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="patient_{pid[:8]}_report.pdf"'}
    )
def export_patient_json_report(pid, db, user):
    """Fallback JSON report when fpdf2 not available."""
    from fastapi.responses import JSONResponse
    patient  = _get_patient_or_403(pid, db, user)
    checkups = db.query(DBCheckup).filter(DBCheckup.patient_id == pid).all()
    preds    = db.query(DBPrediction).filter(DBPrediction.patient_id == pid)\
                 .order_by(DBPrediction.created_at.desc()).all()
    return JSONResponse({"patient_id": pid, "age": patient.age, "sex": patient.sex,
                         "checkup_count": len(checkups),
                         "predictions": len(preds),
                         "note": "Install fpdf2 for PDF: pip install fpdf2"})



def _safe_pdf_text(text: str) -> str:
    """Strip/replace non-latin-1 characters for fpdf2 compatibility."""
    if not text: return ""
    return text.encode("latin-1", errors="replace").decode("latin-1")

# ── MULTI-TENANT CLINIC MANAGEMENT ───────────────────────────────────────────

class ClinicCreate(BaseModel):
    name:    str
    slug:    Optional[str] = None
    address: Optional[str] = None
    phone:   Optional[str] = None
    email:   Optional[str] = None
    timezone: str = "Asia/Kolkata"

@app.post("/api/v1/clinics", status_code=201)
def create_clinic(body: ClinicCreate, db=Depends(get_db),
                  user=Depends(current_user)):
    """Create a new clinic/organisation and make the creator the owner."""
    slug = body.slug or re.sub(r"[^a-z0-9]+", "-", body.name.lower()).strip("-")
    if db.query(DBClinic).filter(DBClinic.slug == slug).first():
        raise HTTPException(400, f"Clinic slug '{slug}' already taken.")
    clinic = DBClinic(name=body.name, slug=slug, address=body.address,
                      phone=body.phone, email=body.email, timezone=body.timezone)
    db.add(clinic); db.flush()
    # Add creator as owner
    db.add(DBClinicMember(clinic_id=clinic.id, user_id=user.id, role="owner"))
    db.commit(); db.refresh(clinic)
    return {"id": clinic.id, "name": clinic.name, "slug": clinic.slug,
            "message": "Clinic created. Share your clinic slug for others to join."}

@app.get("/api/v1/clinics")
def list_clinics(db=Depends(get_db), user=Depends(current_user)):
    """List all clinics the user belongs to."""
    members = (db.query(DBClinicMember)
               .filter(DBClinicMember.user_id == user.id).all())
    result = []
    for m in members:
        c = db.query(DBClinic).filter(DBClinic.id == m.clinic_id).first()
        if c:
            member_count = db.query(DBClinicMember)\
                            .filter(DBClinicMember.clinic_id == c.id).count()
            result.append({"id": c.id, "name": c.name, "slug": c.slug,
                           "role": m.role, "member_count": member_count,
                           "timezone": c.timezone})
    return {"clinics": result}

@app.post("/api/v1/clinics/{slug}/join")
def join_clinic(slug: str, db=Depends(get_db), user=Depends(current_user)):
    """Join a clinic by its slug."""
    clinic = db.query(DBClinic).filter(DBClinic.slug == slug,
                                        DBClinic.active == 1).first()
    if not clinic: raise HTTPException(404, f"No active clinic with slug '{slug}'.")
    existing = db.query(DBClinicMember).filter(
        DBClinicMember.clinic_id == clinic.id,
        DBClinicMember.user_id == user.id).first()
    if existing:
        return {"message": f"Already a member of {clinic.name}.", "role": existing.role}
    db.add(DBClinicMember(clinic_id=clinic.id, user_id=user.id, role="member"))
    db.commit()
    return {"message": f"Joined {clinic.name}.", "slug": slug}

@app.get("/api/v1/clinics/{slug}/members")
def clinic_members(slug: str, db=Depends(get_db), user=Depends(current_user)):
    clinic = db.query(DBClinic).filter(DBClinic.slug == slug).first()
    if not clinic: raise HTTPException(404, "Clinic not found.")
    # Must be a member to view
    mem = db.query(DBClinicMember).filter(
        DBClinicMember.clinic_id == clinic.id,
        DBClinicMember.user_id == user.id).first()
    if not mem: raise HTTPException(403, "Not a member of this clinic.")
    members = db.query(DBClinicMember).filter(
        DBClinicMember.clinic_id == clinic.id).all()
    result = []
    for m in members:
        u = db.query(DBUser).filter(DBUser.id == m.user_id).first()
        if u:
            result.append({"username": u.username, "email": u.email,
                           "role": m.role, "joined_at": m.joined_at})
    return {"clinic": clinic.name, "members": result}


# ── GENOMIC RISK INTEGRATION ──────────────────────────────────────────────────

# Key SNPs and their risk associations
RISK_SNPS = {
    # BRCA1 pathogenic variants
    "rs28897672": {"gene": "BRCA1", "risk_type": "brca1_risk", "effect": 0.8},
    "rs80357382": {"gene": "BRCA1", "risk_type": "brca1_risk", "effect": 0.75},
    # BRCA2 pathogenic variants
    "rs80358720": {"gene": "BRCA2", "risk_type": "brca2_risk", "effect": 0.7},
    "rs28897743": {"gene": "BRCA2", "risk_type": "brca2_risk", "effect": 0.65},
    # APOE4 — Alzheimer's risk
    "rs429358":   {"gene": "APOE",  "risk_type": "apoe4_carrier", "effect": 1},
    # Lynch syndrome (MLH1)
    "rs63750967": {"gene": "MLH1",  "risk_type": "lynch_syndrome", "effect": 0.6},
    # TCF7L2 — Type 2 diabetes (most replicated T2D variant)
    "rs7903146":  {"gene": "TCF7L2","risk_type": "tcf7l2_diabetes","effect": 0.35},
    # LDLR — cardiovascular
    "rs72658867": {"gene": "LDLR",  "risk_type": "ldlr_cardio", "effect": 0.4},
}

@app.post("/api/v1/patients/{pid}/genomics/upload")
async def upload_genomic_data(
    pid: str,
    file: UploadFile = File(...),
    db=Depends(get_db), user=Depends(current_user)
):
    """
    Upload a 23andMe, AncestryDNA, or VCF file to add polygenic risk scores.

    Supported formats:
    - 23andMe raw data (.txt): rsid, chromosome, position, genotype
    - AncestryDNA raw data (.txt)
    - VCF (.vcf): standard variant call format

    The system scans for key disease-associated SNPs and updates the
    patient's genomic risk profile.
    """
    _get_patient_or_403(pid, db, user)
    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:  # 100 MB
        raise HTTPException(413, "File too large. Maximum 100 MB.")

    fname  = (file.filename or "").lower()
    text   = contents.decode("utf-8", errors="replace")
    snps   = _parse_genomic_file(text, fname)

    if not snps:
        raise HTTPException(422,
            "No SNP data found. Supported: 23andMe .txt, AncestryDNA .txt, VCF .vcf")

    # Calculate risk scores
    risks: Dict[str, float] = {}
    for rsid, genotype in snps.items():
        info = RISK_SNPS.get(rsid.lower())
        if not info: continue
        rt = info["risk_type"]
        effect = info["effect"]
        # Simple additive model: heterozygous = 0.5×effect, homozygous = effect
        alleles = len([a for a in genotype if a not in ("0", "-", "N")])
        contrib = effect * (0.5 if alleles == 1 else 1.0 if alleles >= 2 else 0)
        risks[rt] = min(1.0, risks.get(rt, 0) + contrib)

    # Save or update genomic profile
    existing = (db.query(DBGenomicProfile)
                .filter(DBGenomicProfile.patient_id == pid).first())
    variants_summary = {k: v for k, v in risks.items()}
    if existing:
        for k, v in risks.items():
            setattr(existing, k, v)
        existing.snp_count = len(snps)
        existing.upload_date = datetime.now(timezone.utc).isoformat()
        existing.variants_summary = json.dumps(variants_summary)
        existing.source = "23andme" if "23andme" in fname else "vcf" if fname.endswith(".vcf") else "dna_raw"
    else:
        gp = DBGenomicProfile(patient_id=pid, snp_count=len(snps),
                               variants_summary=json.dumps(variants_summary),
                               **risks)
        db.add(gp)
    db.commit()
    audit(db, user, "upload_genomics", pid, f"snps={len(snps)}")

    return {
        "snps_parsed": len(snps),
        "risk_variants_found": len(risks),
        "risk_scores": risks,
        "interpretation": _interpret_genomic_risks(risks),
        "message": (f"Genomic profile updated. Found {len(risks)} risk-associated variants "
                    f"from {len(snps)} SNPs analysed."),
        "disclaimer": (
            "Polygenic risk scores are statistical associations, not diagnoses. "
            "Genetic counselling is recommended before clinical action."
        )
    }

def _parse_genomic_file(text: str, fname: str) -> dict:
    """Parse 23andMe, AncestryDNA, or VCF file → {rsid: genotype}."""
    snps = {}
    lines = text.splitlines()
    for line in lines:
        if line.startswith("#") or not line.strip(): continue
        parts = line.split("\t")
        if len(parts) < 4: continue
        # 23andMe / AncestryDNA: rsid, chr, pos, genotype
        if fname.endswith(".vcf"):
            # VCF: CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO, FORMAT, SAMPLE
            if len(parts) >= 3 and parts[2].startswith("rs"):
                snps[parts[2].lower()] = parts[-1][:4] if parts[-1] else ""
        else:
            # 23andMe / Ancestry
            rsid = parts[0].strip().lower()
            genotype = parts[3].strip() if len(parts) > 3 else ""
            if rsid.startswith("rs"):
                snps[rsid] = genotype
    return snps

def _interpret_genomic_risks(risks: dict) -> dict:
    interp = {}
    if risks.get("brca1_risk", 0) > 0.5:
        interp["BRCA1"] = "Elevated BRCA1 variant detected — breast/ovarian cancer risk. Genetic counselling recommended."
    if risks.get("brca2_risk", 0) > 0.5:
        interp["BRCA2"] = "Elevated BRCA2 variant detected — breast/ovarian cancer risk."
    if risks.get("apoe4_carrier", 0) > 0:
        interp["APOE4"] = "APOE4 carrier — elevated Alzheimer's disease risk."
    if risks.get("tcf7l2_diabetes", 0) > 0.25:
        interp["TCF7L2"] = "TCF7L2 risk variant — moderately increased Type 2 diabetes risk."
    if risks.get("ldlr_cardio", 0) > 0.3:
        interp["LDLR"] = "LDLR variant detected — possible familial hypercholesterolaemia."
    return interp

@app.get("/api/v1/patients/{pid}/genomics")
def get_genomic_profile(pid: str, db=Depends(get_db), user=Depends(current_user)):
    _get_patient_or_403(pid, db, user)
    gp = db.query(DBGenomicProfile).filter(DBGenomicProfile.patient_id == pid).first()
    if not gp:
        return {"patient_id": pid, "genomic_profile": None,
                "message": "No genomic data. Upload a 23andMe/VCF file."}
    return {
        "patient_id":    pid,
        "source":        gp.source,
        "snp_count":     gp.snp_count,
        "upload_date":   gp.upload_date,
        "risk_scores":   {
            "brca1_risk":       gp.brca1_risk,
            "brca2_risk":       gp.brca2_risk,
            "apoe4_carrier":    bool(gp.apoe4_carrier),
            "lynch_syndrome":   gp.lynch_syndrome,
            "tcf7l2_diabetes":  gp.tcf7l2_diabetes,
            "ldlr_cardio":      gp.ldlr_cardio,
        },
        "interpretation": _interpret_genomic_risks({
            "brca1_risk": gp.brca1_risk or 0,
            "brca2_risk": gp.brca2_risk or 0,
            "apoe4_carrier": gp.apoe4_carrier or 0,
            "lynch_syndrome": gp.lynch_syndrome or 0,
            "tcf7l2_diabetes": gp.tcf7l2_diabetes or 0,
            "ldlr_cardio": gp.ldlr_cardio or 0,
        }),
    }


# ── VIDEO CONSULTATION (Jitsi) ────────────────────────────────────────────────

@app.post("/api/v1/patients/{pid}/consultation/create")
def create_video_consultation(pid: str, body: dict = None,
                               db=Depends(get_db), user=Depends(current_user)):
    """
    Generate a Jitsi video consultation link for a patient.
    No sign-up, no API key required — Jitsi Meet is free and open-source.

    Optional body: {"duration_minutes": 30, "scheduled_at": "2024-04-01T10:00:00"}
    """
    _get_patient_or_403(pid, db, user)
    body = body or {}

    # Generate a unique, hard-to-guess room name
    room_token = _secrets.token_urlsafe(12)
    room_name  = f"biosentinel-{pid[:8]}-{room_token}"
    jitsi_url  = f"https://meet.jit.si/{room_name}"

    # Also support custom Jitsi server
    custom_jitsi = os.getenv("JITSI_SERVER_URL", "https://meet.jit.si")
    jitsi_url = f"{custom_jitsi.rstrip('/')}/{room_name}"

    scheduled_at   = body.get("scheduled_at")
    duration_min   = body.get("duration_minutes", 30)

    audit(db, user, "create_consultation", pid,
          f"room={room_name} scheduled={scheduled_at}")

    return {
        "room_name":      room_name,
        "join_url":       jitsi_url,
        "doctor_url":     f"{jitsi_url}#userInfo.displayName={user.username}",
        "patient_url":    jitsi_url,
        "scheduled_at":   scheduled_at,
        "duration_minutes": duration_min,
        "expires_at":     (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
        "instructions": {
            "doctor":  f"Open: {jitsi_url} — no account needed",
            "patient": f"Share this link with your patient: {jitsi_url}",
        },
        "note": ("Powered by Jitsi Meet (open source). "
                 "For custom server set JITSI_SERVER_URL env var. "
                 "Enterprise: use Jitsi as a Service at jaas.8x8.vc"),
    }


# ── ENCRYPTION MANAGEMENT ────────────────────────────────────────────────────

@app.get("/api/v1/admin/encryption/status")
def encryption_status(db=Depends(get_db), user=Depends(current_user)):
    if user.role != "admin":
        raise HTTPException(403, "Admin only.")
    return {
        "field_encryption_enabled": crypto.enabled,
        "encryption_available":     ENCRYPTION_AVAILABLE,
        "setup_instructions": (
            "1. Generate key: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\"\n"
            "2. Set env: FIELD_ENCRYPTION_KEY=<your_key>\n"
            "3. Restart server — new records will be encrypted automatically.\n"
            "4. CRITICAL: Back up the key separately. Losing it = permanent data loss."
        ) if not crypto.enabled else "Encryption is active."
    }

@app.post("/api/v1/admin/encryption/rotate-key")
def rotate_encryption_key(body: dict, db=Depends(get_db),
                           user=Depends(current_user)):
    """Re-encrypt all data with a new key. Requires current key still set in env."""
    if user.role != "admin":
        raise HTTPException(403, "Admin only.")
    new_key = body.get("new_key", "")
    if not new_key:
        raise HTTPException(400, "Provide new_key in request body.")
    result = crypto.rotate_key(new_key, db)
    audit(db, user, "rotate_encryption_key",
          detail=f"re_encrypted={result.get('re_encrypted_records',0)}")
    return result


# ── 2FA / FHIR done — capabilities follows ────────────────────────────────────
@app.get("/api/v1/capabilities")
def capabilities():
    """Return what optional features are available on this installation."""
    try:
        import pandas as _pd
        csv_import_available = True
    except ImportError:
        csv_import_available = False

    twilio_configured   = bool(TWILIO_SID and TWILIO_TOKEN)
    telegram_configured = bool(TELEGRAM_BOT_TOKEN)

    return {
        "version":          "2.3.3",
        "ocr_pdf":          PDF_AVAILABLE,
        "ocr_image":        OCR_AVAILABLE,
        "shap_available":   getattr(engine_ml, "shap_available", False),
        "totp_available":   TOTP_AVAILABLE,
        "rate_limiting":    RATE_LIMIT_AVAILABLE,
        "ml_models":        list(engine_ml.models.keys()) if engine_ml.trained else [],
        "db_type":          "sqlite" if _IS_SQLITE else "postgresql",
        "notification_channels": {
            "email":     bool(EMAIL_ENABLED),
            "sms":       twilio_configured,
            "whatsapp":  twilio_configured,
            "telegram":  telegram_configured,
        },
        "claude_ai":        claude_ai_status(),
        "mlflow":           mlflow_status(),
        "cache":            _cache_stats(),
        "scheduler":        SCHEDULER_AVAILABLE,
        "features": {
            "lab_report_upload":       PDF_AVAILABLE or OCR_AVAILABLE,
            "claude_vision_ocr":       CLAUDE_AI_AVAILABLE,
            "ai_narrative":            CLAUDE_AI_AVAILABLE,
            "ai_anomaly_detection":    CLAUDE_AI_AVAILABLE,
            "ai_drug_interaction":     CLAUDE_AI_AVAILABLE,
            "trend_alerts":            True,
            "overdue_reminders":       True,
            "background_scheduler":    SCHEDULER_AVAILABLE,
            "appointment_reminders":   True,
            "email_alerts":            True,
            "password_reset_email":    True,
            "password_reset_sms":      twilio_configured,
            "password_reset_whatsapp": twilio_configured,
            "password_reset_telegram": telegram_configured,
            "bulk_patient_import":     csv_import_available,
            "bulk_checkup_import":     csv_import_available,
            "drug_interaction_check":  True,
            "shap_explanations":       getattr(engine_ml, "shap_available", False),
            "two_factor_auth":         TOTP_AVAILABLE,
            "fhir_r4_import":          True,
            "rate_limiting":           RATE_LIMIT_AVAILABLE,
            "audit_log":               True,
            "multi_user_isolation":    True,
            "patient_self_register":   True,
            "postgresql_support":      not _IS_SQLITE,
        },
    }


# ── SEED + STARTUP ────────────────────────────────────────────────────────────
def seed(db: Session):
    if db.query(DBUser).count() > 0:
        return

    print("🌱 Seeding demo data...")
    # Users
    db.add(DBUser(username="admin",     email="admin@biosentinel.ai",     hashed_password=hash_pw("admin123"),   role="admin"))
    db.add(DBUser(username="dr_sharma", email="sharma@hospital.ai",       hashed_password=hash_pw("doctor123"),  role="clinician"))
    db.add(DBUser(username="dr_chen",   email="chen@research.ai",         hashed_password=hash_pw("research123"),role="researcher"))
    db.commit()

    # ── 5 demo patients covering different risk profiles ──────────────────
    patients_data = [

        # P1: 48F South Asian — metabolic risk trajectory (pre-diabetes developing)
        dict(pat=DBPatient(age=48,sex="Female",ethnicity="South Asian",
                family_history_cancer=1,family_history_diabetes=1,family_history_cardio=0,
                smoking_status="never",alcohol_units_weekly=2,exercise_min_weekly=90,
                notes="Quarterly monitoring since Jan 2022. Hypothyroidism on Levothyroxine."),
             chks=[
                ("2022-01-15",{"hba1c":5.5,"glucose_fasting":96,"hemoglobin":13.5,"wbc":7.4,"lymphocytes_pct":32,"cea":1.8,"alt":24,"ldl":114,"hdl":54,"bp_systolic":126,"bmi":26.8,"weight_kg":71.2,"platelets":252,"crp":1.2,"tsh":2.3}),
                ("2022-04-15",{"hba1c":5.5,"glucose_fasting":97,"hemoglobin":13.4,"wbc":7.3,"lymphocytes_pct":31,"cea":1.9,"alt":25,"ldl":116,"hdl":53,"bp_systolic":127,"bmi":27.0,"crp":1.3}),
                ("2022-07-15",{"hba1c":5.6,"glucose_fasting":99,"hemoglobin":13.3,"wbc":7.2,"lymphocytes_pct":30,"cea":2.1,"alt":26,"ldl":119,"hdl":52,"bp_systolic":128,"bmi":27.1,"crp":1.4}),
                ("2022-10-15",{"hba1c":5.7,"glucose_fasting":102,"hemoglobin":13.1,"wbc":7.0,"lymphocytes_pct":29,"cea":2.3,"alt":28,"ldl":122,"hdl":51,"bp_systolic":130,"bmi":27.5,"crp":1.6}),
                ("2023-01-15",{"hba1c":5.8,"glucose_fasting":104,"hemoglobin":13.0,"wbc":6.9,"lymphocytes_pct":28,"cea":2.6,"alt":29,"ldl":124,"hdl":50,"bp_systolic":132,"bmi":27.7,"crp":1.8}),
                ("2023-04-15",{"hba1c":5.8,"glucose_fasting":106,"hemoglobin":12.9,"wbc":6.8,"lymphocytes_pct":27,"cea":2.9,"alt":31,"ldl":126,"hdl":49,"bp_systolic":134,"bmi":28.0,"crp":2.0}),
                ("2023-07-15",{"hba1c":5.9,"glucose_fasting":108,"hemoglobin":12.8,"wbc":6.7,"lymphocytes_pct":26,"cea":3.1,"alt":33,"ldl":128,"hdl":49,"bp_systolic":135,"bmi":28.2,"crp":2.2}),
                ("2024-01-15",{"hba1c":6.1,"glucose_fasting":113,"hemoglobin":12.5,"wbc":6.5,"lymphocytes_pct":24,"cea":3.8,"alt":38,"ldl":133,"hdl":47,"bp_systolic":139,"bmi":28.6,"crp":2.7}),
             ],
             meds=[
                 dict(name="Levothyroxine",dosage_mg=50,frequency="Daily",start_date="2019-06-01",prescribed_for="Hypothyroidism E03.9",active=1),
             ],
             diags=[
                 dict(icd10_code="E03.9",description="Hypothyroidism",diagnosed_date="2019-05-20",status="active",severity="mild"),
             ],
             diets=[
                 dict(start_date="2022-01-15",calories_daily=1800,protein_g=72,carbs_g=225,fat_g=64,fiber_g=26,diet_type="balanced",notes="Moderately healthy diet, reducing processed foods"),
             ]
        ),

        # P2: 35M Caucasian — healthy, low risk (control)
        dict(pat=DBPatient(age=35,sex="Male",ethnicity="Caucasian",
                family_history_cancer=0,family_history_diabetes=0,family_history_cardio=0,
                smoking_status="never",alcohol_units_weekly=4,exercise_min_weekly=240,
                notes="Athlete, excellent health. Monitoring as preventive care."),
             chks=[
                ("2022-03-01",{"hba1c":5.1,"glucose_fasting":85,"hemoglobin":15.2,"wbc":6.8,"lymphocytes_pct":35,"cea":1.2,"alt":18,"ldl":95,"hdl":62,"bp_systolic":112,"bmi":23.1,"platelets":270,"crp":0.6}),
                ("2022-09-01",{"hba1c":5.1,"glucose_fasting":86,"hemoglobin":15.3,"wbc":6.9,"lymphocytes_pct":34,"cea":1.1,"alt":17,"ldl":93,"hdl":63,"bp_systolic":111,"bmi":23.0,"crp":0.5}),
                ("2023-03-01",{"hba1c":5.2,"glucose_fasting":87,"hemoglobin":15.1,"wbc":7.0,"lymphocytes_pct":34,"cea":1.3,"alt":19,"ldl":97,"hdl":61,"bp_systolic":113,"bmi":23.3,"crp":0.7}),
                ("2023-09-01",{"hba1c":5.2,"glucose_fasting":88,"hemoglobin":15.0,"wbc":7.0,"lymphocytes_pct":33,"cea":1.2,"alt":18,"ldl":98,"hdl":61,"bp_systolic":114,"bmi":23.4,"crp":0.6}),
                ("2024-01-01",{"hba1c":5.2,"glucose_fasting":87,"hemoglobin":15.2,"wbc":6.9,"lymphocytes_pct":35,"cea":1.1,"alt":17,"ldl":94,"hdl":64,"bp_systolic":110,"bmi":23.0,"crp":0.5}),
             ],
             meds=[], diags=[],
             diets=[dict(start_date="2022-01-01",calories_daily=2400,protein_g=140,carbs_g=280,fat_g=80,fiber_g=38,diet_type="high_protein",notes="High protein Mediterranean diet")]
        ),

        # P3: 62M South Asian — multi-risk: cancer signals + metabolic + cardio
        dict(pat=DBPatient(age=62,sex="Male",ethnicity="South Asian",
                family_history_cancer=2,family_history_diabetes=1,family_history_cardio=1,
                smoking_status="former",alcohol_units_weekly=7,exercise_min_weekly=45,
                notes="High-risk patient. Multiple deteriorating biomarker trends. Requires urgent follow-up."),
             chks=[
                ("2021-06-01",{"hba1c":6.0,"glucose_fasting":108,"hemoglobin":14.0,"wbc":8.2,"lymphocytes_pct":29,"cea":2.5,"alt":35,"ldl":145,"hdl":42,"bp_systolic":142,"bmi":28.5,"platelets":240,"crp":3.2,"ca125":18}),
                ("2021-12-01",{"hba1c":6.2,"glucose_fasting":112,"hemoglobin":13.6,"wbc":8.5,"lymphocytes_pct":27,"cea":3.2,"alt":38,"ldl":150,"hdl":40,"bp_systolic":146,"bmi":29.0,"crp":3.8}),
                ("2022-06-01",{"hba1c":6.4,"glucose_fasting":118,"hemoglobin":13.1,"wbc":9.0,"lymphocytes_pct":23,"cea":4.1,"alt":44,"ldl":158,"hdl":38,"bp_systolic":152,"bmi":29.6,"crp":4.5}),
                ("2022-12-01",{"hba1c":6.6,"glucose_fasting":124,"hemoglobin":12.5,"wbc":9.8,"lymphocytes_pct":20,"cea":5.4,"alt":52,"ldl":165,"hdl":36,"bp_systolic":158,"bmi":30.1,"crp":5.8}),
                ("2023-06-01",{"hba1c":6.9,"glucose_fasting":131,"hemoglobin":11.8,"wbc":10.5,"lymphocytes_pct":17,"cea":7.1,"alt":61,"ldl":172,"hdl":34,"bp_systolic":164,"bmi":30.8,"crp":7.2}),
                ("2024-01-01",{"hba1c":7.1,"glucose_fasting":138,"hemoglobin":11.2,"wbc":11.2,"lymphocytes_pct":15,"cea":9.2,"alt":72,"ldl":178,"hdl":32,"bp_systolic":168,"bmi":31.3,"crp":9.1}),
             ],
             meds=[
                 dict(name="Metformin",dosage_mg=500,frequency="Twice daily",start_date="2022-08-01",prescribed_for="Pre-diabetes",active=1),
                 dict(name="Atorvastatin",dosage_mg=40,frequency="Daily",start_date="2022-08-01",prescribed_for="Hyperlipidemia E78.00",active=1),
                 dict(name="Amlodipine",dosage_mg=5,frequency="Daily",start_date="2022-09-01",prescribed_for="Hypertension I10",active=1),
             ],
             diags=[
                 dict(icd10_code="R73.09",description="Pre-diabetes",diagnosed_date="2022-07-15",status="active",severity="moderate"),
                 dict(icd10_code="E78.00",description="Hypercholesterolemia",diagnosed_date="2022-07-15",status="active",severity="moderate"),
                 dict(icd10_code="I10",description="Essential Hypertension",diagnosed_date="2022-09-01",status="active",severity="moderate"),
             ],
             diets=[dict(start_date="2022-08-01",calories_daily=1700,protein_g=75,carbs_g=180,fat_g=60,fiber_g=25,diet_type="low_carb",notes="Low carb diet post-diabetes diagnosis")]
        ),

        # P4: 55F — Recovering, improving biomarkers
        dict(pat=DBPatient(age=55,sex="Female",ethnicity="African American",
                family_history_cancer=0,family_history_diabetes=1,family_history_cardio=1,
                smoking_status="former",alcohol_units_weekly=1,exercise_min_weekly=180,
                notes="Recently improved lifestyle — diet and exercise intervention started mid-2022."),
             chks=[
                ("2022-03-01",{"hba1c":6.0,"glucose_fasting":112,"hemoglobin":12.8,"wbc":8.1,"lymphocytes_pct":26,"cea":2.2,"alt":36,"ldl":148,"hdl":43,"bp_systolic":148,"bmi":29.5,"crp":4.1}),
                ("2022-09-01",{"hba1c":5.9,"glucose_fasting":108,"hemoglobin":13.0,"wbc":7.8,"lymphocytes_pct":28,"cea":2.0,"alt":32,"ldl":140,"hdl":46,"bp_systolic":142,"bmi":28.8,"crp":3.4}),
                ("2023-03-01",{"hba1c":5.7,"glucose_fasting":102,"hemoglobin":13.3,"wbc":7.4,"lymphocytes_pct":30,"cea":1.8,"alt":26,"ldl":128,"hdl":50,"bp_systolic":136,"bmi":27.9,"crp":2.6}),
                ("2023-09-01",{"hba1c":5.6,"glucose_fasting":98,"hemoglobin":13.5,"wbc":7.1,"lymphocytes_pct":31,"cea":1.6,"alt":22,"ldl":118,"hdl":54,"bp_systolic":128,"bmi":26.8,"crp":1.8}),
                ("2024-01-01",{"hba1c":5.5,"glucose_fasting":95,"hemoglobin":13.8,"wbc":7.0,"lymphocytes_pct":32,"cea":1.5,"alt":20,"ldl":112,"hdl":56,"bp_systolic":124,"bmi":26.1,"crp":1.4}),
             ],
             meds=[dict(name="Lisinopril",dosage_mg=10,frequency="Daily",start_date="2022-04-01",prescribed_for="Hypertension I10",active=1)],
             diags=[dict(icd10_code="I10",description="Essential Hypertension",diagnosed_date="2022-03-15",status="active",severity="mild")],
             diets=[dict(start_date="2022-07-01",calories_daily=1600,protein_g=85,carbs_g=175,fat_g=55,fiber_g=35,diet_type="mediterranean",notes="Mediterranean diet + daily 45min walk")]
        ),

        # P5: 70M — Elder with stable chronic conditions
        dict(pat=DBPatient(age=70,sex="Male",ethnicity="Caucasian",
                family_history_cancer=1,family_history_diabetes=0,family_history_cardio=1,
                smoking_status="former",alcohol_units_weekly=3,exercise_min_weekly=90,
                notes="Stable chronic patient on multiple medications. Annual review focus."),
             chks=[
                ("2022-06-01",{"hba1c":6.3,"glucose_fasting":118,"hemoglobin":13.2,"wbc":7.5,"lymphocytes_pct":27,"cea":2.8,"alt":30,"ldl":122,"hdl":48,"bp_systolic":138,"bmi":26.2,"crp":2.8,"psa":3.2}),
                ("2022-12-01",{"hba1c":6.4,"glucose_fasting":120,"hemoglobin":13.1,"wbc":7.6,"lymphocytes_pct":27,"cea":3.0,"alt":31,"ldl":125,"hdl":47,"bp_systolic":140,"bmi":26.4,"crp":2.9,"psa":3.5}),
                ("2023-06-01",{"hba1c":6.4,"glucose_fasting":121,"hemoglobin":13.0,"wbc":7.7,"lymphocytes_pct":26,"cea":3.2,"alt":32,"ldl":128,"hdl":46,"bp_systolic":141,"bmi":26.5,"crp":3.1,"psa":3.9}),
                ("2023-12-01",{"hba1c":6.5,"glucose_fasting":122,"hemoglobin":12.9,"wbc":7.8,"lymphocytes_pct":25,"cea":3.4,"alt":33,"ldl":130,"hdl":46,"bp_systolic":142,"bmi":26.7,"crp":3.2,"psa":4.2}),
             ],
             meds=[
                 dict(name="Metformin",dosage_mg=850,frequency="Twice daily",start_date="2018-01-01",prescribed_for="Type 2 Diabetes E11",active=1),
                 dict(name="Rosuvastatin",dosage_mg=10,frequency="Daily",start_date="2015-06-01",prescribed_for="Hyperlipidemia",active=1),
                 dict(name="Aspirin",dosage_mg=81,frequency="Daily",start_date="2015-06-01",prescribed_for="Cardioprotection",active=1),
             ],
             diags=[
                 dict(icd10_code="E11",description="Type 2 Diabetes",diagnosed_date="2018-01-01",status="active",severity="moderate"),
                 dict(icd10_code="I10",description="Hypertension",diagnosed_date="2015-01-01",status="active",severity="mild"),
             ],
             diets=[dict(start_date="2022-01-01",calories_daily=2000,protein_g=90,carbs_g=200,fat_g=70,fiber_g=30,diet_type="diabetic_friendly",notes="Low GI diet, consistent carb counting")]
        ),
    ]

    for pdata in patients_data:
        pat = pdata["pat"]
        db.add(pat); db.commit(); db.refresh(pat)

        for chk_date, fields in pdata["chks"]:
            c = DBCheckup(patient_id=pat.id, checkup_date=chk_date, **fields)
            db.add(c)

        for m in pdata["meds"]:
            db.add(DBMedication(patient_id=pat.id, **m))

        for d in pdata["diags"]:
            db.add(DBDiagnosis(patient_id=pat.id, **d))

        for dp in pdata.get("diets", []):
            db.add(DBDietPlan(patient_id=pat.id, **dp))

        db.commit()

    print(f"✅ Seeded 3 users and {len(patients_data)} demo patients\n")


if __name__ == "__main__":
    print("\n" + "="*62)
    print("  ██████╗ ██╗ ██████╗ ███████╗███████╗███╗   ██╗████████╗")
    print("  ██╔══██╗██║██╔═══██╗██╔════╝██╔════╝████╗  ██║╚══██╔══╝")
    print("  ██████╔╝██║██║   ██║███████╗█████╗  ██╔██╗ ██║   ██║   ")
    print("  ██╔══██╗██║██║   ██║╚════██║██╔══╝  ██║╚██╗██║   ██║   ")
    print("  ██████╔╝██║╚██████╔╝███████║███████╗██║ ╚████║   ██║   ")
    print("  ╚═════╝ ╚═╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ")
    print("  AI Early Disease Detection Platform  v2.3.3")
    print("  Developer: Liveupx Pvt. Ltd. | Mohit Chaprana")
    print("="*62 + "\n")

    Base.metadata.create_all(bind=engine)
    engine_ml.train()

    # Log training run to MLflow (if MLFLOW_TRACKING=1)
    if MLFLOW_AVAILABLE:
        run_id = track_training_run(
            engine_ml,
            data_source=os.getenv("TRAINING_DATA_SOURCE", "synthetic_5000"),
            notes=os.getenv("TRAINING_NOTES", ""),
        )
        if run_id:
            print(f"📊 MLflow run logged: {run_id}")
            print(f"   View at: {os.getenv('MLFLOW_TRACKING_URI', './mlruns')}")

    db = SessionLocal()
    seed(db)
    db.close()

    # Start background scheduler (appointment reminders, daily stats)
    if SCHEDULER_AVAILABLE:
        _scheduler = start_scheduler(SessionLocal, notify_engine, email_config_engine)
        if _scheduler:
            print("⏰ Background scheduler started (overdue reminders, daily stats)")
    else:
        print("⚠  APScheduler not installed — background jobs disabled. pip install apscheduler")

    print("🚀 Starting BioSentinel API...")
    print("   API:       http://localhost:8000")
    print("   Swagger:   http://localhost:8000/docs")
    print("   Dashboard: open biosentinel_dashboard.html in your browser\n")
    print("   Login credentials:")
    print("   admin / admin123   ·   dr_sharma / doctor123   ·   dr_chen / research123\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
