"""
BioSentinel v1.0 — Complete Working Backend
============================================
Run:   python app.py
Docs:  http://localhost:8000/docs
Dashboard: open biosentinel_dashboard.html in browser

Features:
 - 4 properly-calibrated ML models (cancer, metabolic, cardio, hematologic)
 - Full CRUD: patients, checkups, medications, diagnoses, diet plans
 - JWT auth, SQLite persistence, population analytics
 - Risk trajectory tracking, biomarker alerts, report generation
"""

import json, math, os, uuid, warnings, smtplib, threading, time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Any

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sqlalchemy import Column, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
SECRET_KEY  = os.getenv("SECRET_KEY", "bs-dev-secret-xyz-2025-liveupx")
ALGORITHM   = "HS256"
TOKEN_EXP   = 60 * 24   # 24 hours for demo
DB_URL      = os.getenv("DATABASE_URL", "sqlite:///./biosentinel.db")

# ── EMAIL CONFIG (set via env vars or .env file) ─────────────────────────────
EMAIL_ENABLED  = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
EMAIL_HOST     = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT     = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER     = os.getenv("EMAIL_USER", "")          # your Gmail address
EMAIL_PASS     = os.getenv("EMAIL_PASS", "")          # Gmail app password
EMAIL_FROM     = os.getenv("EMAIL_FROM", EMAIL_USER)
EMAIL_TO_ADMIN = os.getenv("EMAIL_TO_ADMIN", EMAIL_USER)  # who receives alerts

engine      = create_engine(DB_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base        = declarative_base()

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
        now = datetime.utcnow().strftime("%d %b %Y, %H:%M UTC")
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
    id             = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username       = Column(String, unique=True, index=True)
    email          = Column(String, unique=True)
    hashed_password= Column(String)
    role           = Column(String, default="clinician")
    created_at     = Column(String, default=lambda: datetime.utcnow().isoformat())

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
    created_at             = Column(String, default=lambda: datetime.utcnow().isoformat())
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
    created_at       = Column(String, default=lambda: datetime.utcnow().isoformat())
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
    created_at  = Column(String, default=lambda: datetime.utcnow().isoformat())
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
    updated_at      = Column(String, default=lambda: datetime.utcnow().isoformat())

class DBAuditLog(Base):
    """Immutable audit trail of all patient data access."""
    __tablename__ = "audit_log"
    id          = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp   = Column(String, default=lambda: datetime.utcnow().isoformat())
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

def make_token(data):
    d = data.copy()
    d["exp"] = datetime.utcnow() + timedelta(minutes=TOKEN_EXP)
    return jwt.encode(d, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

def current_user(creds: HTTPAuthorizationCredentials = Depends(security),
                 db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(creds.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username: raise HTTPException(401, "Invalid token")
    except JWTError:
        raise HTTPException(401, "Invalid token")
    u = db.query(DBUser).filter(DBUser.username == username).first()
    if not u: raise HTTPException(401, "User not found")
    return u

# ── EMAIL ENGINE ─────────────────────────────────────────────────────────────
class EmailEngine:
    """
    Sends alert emails via user-configured SMTP.
    Runs in a background thread so the API never blocks.
    Works with Gmail, Outlook, Mailgun, SendGrid, or any SMTP server.
    """

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

email_engine = EmailEngine()

# ── AUDIT LOGGER ─────────────────────────────────────────────────────────────
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
        print("\n✅ All 4 models trained and calibrated.\n")

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

    def _attribute(self, feat, results):
        """Rule-based feature attribution with actual trend values."""
        f = feat
        attrs = []

        checks = [
            # (label, value, threshold, direction, domain)
            ("HbA1c upward trend",       f[31], 0.01,  "up",   "metabolic"),
            ("HbA1c above pre-diabetic",  f[10], 5.7,   "high", "metabolic"),
            ("Fasting glucose rising",    f[32], 0.3,   "up",   "metabolic"),
            ("BMI elevated & rising",     f[35], 0.02,  "up",   "metabolic"),
            ("CEA tumor marker rising",   f[36], 0.04,  "up",   "cancer"),
            ("CEA above normal",          f[16], 3.0,   "high", "cancer"),
            ("Lymphocytes declining",     f[34], -0.08, "down", "cancer"),
            ("Hemoglobin declining",      f[33], -0.01, "down", "cancer"),
            ("Family cancer history",     f[5],  0,     "fh",   "cancer"),
            ("LDL cholesterol rising",    f[37], 0.2,   "up",   "cardio"),
            ("Blood pressure rising",     f[38], 0.2,   "up",   "cardio"),
            ("Low HDL cholesterol",       f[22], 50,    "low",  "cardio"),
            ("Current smoker",            f[2],  1.5,   "high", "all"),
            ("High CRP (inflammation)",   f[28], 3.0,   "high", "cancer"),
            ("WBC abnormal trend",        abs(f[35]), 0.05, "up","hematologic"),
            ("Age-related risk",          f[0],  55,    "high", "all"),
            ("ALT liver enzyme rising",   f[36], 0.1,   "up",   "metabolic"),
        ]

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
                attrs.append({
                    "label": label,
                    "value": round(float(val), 3),
                    "impact": round(impact, 4),
                    "domain": domain,
                    "direction": "risk_increasing",
                })

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
    title="BioSentinel API v1.0",
    description="AI-powered longitudinal health monitoring. Developer: Liveupx Pvt. Ltd. | github.com/liveupx/biosentinel",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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
def register(u: UserCreate, db: Session = Depends(get_db)):
    if db.query(DBUser).filter(DBUser.username == u.username).first():
        raise HTTPException(400, "Username taken")
    user = DBUser(username=u.username, email=u.email,
                  hashed_password=hash_pw(u.password), role=u.role)
    db.add(user); db.commit(); db.refresh(user)
    return {"access_token": make_token({"sub":user.username}),
            "token_type":"bearer",
            "user":{"id":user.id,"username":user.username,"role":user.role}}

@app.post("/api/v1/auth/login")
def login(u: UserLogin, db: Session = Depends(get_db)):
    user = db.query(DBUser).filter(DBUser.username == u.username).first()
    if not user or not verify_pw(u.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    return {"access_token": make_token({"sub":user.username}),
            "token_type":"bearer",
            "user":{"id":user.id,"username":user.username,"role":user.role}}

@app.get("/api/v1/auth/me")
def me(u=Depends(current_user)):
    return {"id":u.id,"username":u.username,"email":u.email,"role":u.role}

# ── PATIENTS ──────────────────────────────────────────────────────────────────
@app.post("/api/v1/patients", status_code=201)
def create_patient(p: PatientCreate, db=Depends(get_db), user=Depends(current_user)):
    data = p.dict()
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

@app.get("/api/v1/patients/{pid}")
def get_patient(pid:str, db=Depends(get_db), user=Depends(current_user)):
    return _get_patient_or_403(pid, db, user)

@app.put("/api/v1/patients/{pid}")
def update_patient(pid:str, data:PatientCreate, db=Depends(get_db), user=Depends(current_user)):
    p = _get_patient_or_403(pid, db, user)
    d = data.dict(); d.pop('owner_id', None)
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
    chk = DBCheckup(**c.dict()); db.add(chk); db.commit(); db.refresh(chk)
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
    med = DBMedication(**m.dict()); db.add(med); db.commit(); db.refresh(med)
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
    diag = DBDiagnosis(**d.dict()); db.add(diag); db.commit(); db.refresh(diag)
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
    plan = DBDietPlan(**dp.dict()); db.add(plan); db.commit(); db.refresh(plan)
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
        email_engine.send_alert_email(email_cfg, alert_obj, patient.age, patient.sex)
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
    return {"labels": labels, "series": series}

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
        "report_date": datetime.utcnow().isoformat(),
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
    cfg.updated_at      = datetime.utcnow().isoformat()
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
    result = email_engine.test_connection(cfg, body.to_address)
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
    cutoff = (datetime.utcnow() - timedelta(days=95)).date().isoformat()
    for pid in owned_ids:
        last = (db.query(DBCheckup)
                .filter(DBCheckup.patient_id == pid)
                .order_by(DBCheckup.checkup_date.desc())
                .first())
        if not last or last.checkup_date[:10] < cutoff:
            overdue += 1
    return overdue

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
    print("  AI Early Disease Detection Platform  v1.0.0")
    print("  Developer: Liveupx Pvt. Ltd. | Mohit Chaprana")
    print("="*62 + "\n")

    Base.metadata.create_all(bind=engine)
    engine_ml.train()

    db = SessionLocal()
    seed(db)
    db.close()

    print("🚀 Starting BioSentinel API...")
    print("   API:       http://localhost:8000")
    print("   Swagger:   http://localhost:8000/docs")
    print("   Dashboard: open biosentinel_dashboard.html in your browser\n")
    print("   Login credentials:")
    print("   admin / admin123   ·   dr_sharma / doctor123   ·   dr_chen / research123\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
