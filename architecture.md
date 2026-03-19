# BioSentinel Architecture (v2.2.0)

## Core principle

**Time is the most valuable dimension in health data.**

Traditional clinical tools analyse a snapshot. BioSentinel analyses the full trajectory — detecting the subtle directional trends that predict disease years before a single test result crosses a threshold.

---

## System layers

```
┌─────────────────────────────────────────────────────────────────┐
│  Interfaces                                                      │
│  ┌─────────────────────┐  ┌───────────────────┐  ┌──────────┐  │
│  │ Clinician Dashboard  │  │  Patient Portal    │  │ REST API │  │
│  │ (dark, Chart.js)     │  │  (light, mobile)   │  │  /docs   │  │
│  └─────────┬───────────┘  └────────┬──────────┘  └────┬─────┘  │
└────────────┼────────────────────────┼────────────────────┼──────┘
             │                        │                    │
┌────────────▼────────────────────────▼────────────────────▼──────┐
│  FastAPI Backend (app.py — 5,200 lines)                          │
│                                                                  │
│  Auth         Patients      Checkups     Predictions             │
│  JWT + 2FA    CRUD          30+ fields   4 ML models             │
│  Rate limit   Ownership     Trends       SHAP explanations       │
│  Audit log    Isolation     OCR import   Risk trajectory         │
│                                                                  │
│  Claude AI    Scheduler     Cache        Analytics               │
│  Vision OCR   Overdue       In-memory    Percentiles             │
│  Narratives   reminders     TTL 5min     Population stats        │
│  Anomalies    Daily stats   Invalidate   Biomarker compare       │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────┐
│  ML Engine (BioSentinelEngine)                                   │
│                                                                  │
│  4× GradientBoostingRegressor + IsotonicRegression calibration  │
│  49 features: demographics, latest values, slopes, volatility   │
│  Real SHAP TreeExplainer (shap library)                          │
│  Training: synthetic 5,000 (dev) → MIMIC-IV 65K (production)    │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────┐
│  Data Layer                                                      │
│  SQLite (dev) ←──── migrate_to_postgres.py ────→ PostgreSQL      │
│  10 tables: users, patients, checkups, predictions, alerts,      │
│             medications, diagnoses, diet_plans, audit_logs,      │
│             genomic_profiles                                     │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key files

| File | Role | Lines |
|---|---|---|
| `app.py` | Complete FastAPI backend — all endpoints, models, business logic | 5,200+ |
| `claude_ai.py` | Claude API integration — Vision OCR, narratives, anomaly detection | 350 |
| `scheduler.py` | APScheduler background jobs — overdue reminders, daily stats | 170 |
| `mlflow_tracking.py` | Optional experiment tracking for model training runs | 210 |
| `train_mimic.py` | MIMIC-IV training script — BigQuery export + local model training | 300 |
| `migrate_to_postgres.py` | One-command SQLite → PostgreSQL migration with verification | 250 |
| `biosentinel_dashboard.html` | Clinician UI — dark theme, 6-tab patient view, AI Insights tab | 1,600+ |
| `biosentinel_patient_portal.html` | Patient self-service — plain-English, mobile-first | 650 |
| `sw.js` | Service worker — offline support, push notifications, background sync | 100 |
| `manifest.json` | PWA manifest — installable on iOS/Android | 60 |

---

## ML models

| Domain | Algorithm | Calibration | Training data |
|---|---|---|---|
| Cancer | GradientBoostingRegressor | IsotonicRegression | Synthetic 5K → MIMIC-IV 65K |
| Metabolic | GradientBoostingRegressor | IsotonicRegression | Synthetic 5K → MIMIC-IV 65K |
| Cardiovascular | GradientBoostingRegressor | IsotonicRegression | Synthetic 5K → MIMIC-IV 65K |
| Hematologic | GradientBoostingRegressor | IsotonicRegression | Synthetic 5K → MIMIC-IV 65K |

**Feature engineering** — 49 features per prediction:
- Patient demographics: age, sex, ethnicity, smoking, alcohol, exercise
- Family history: cancer/diabetes/cardiovascular (count of first-degree relatives)
- Latest biomarker values: 20 fields (HbA1c, CEA, lymphocytes, WBC, LDL, etc.)
- Trend slopes: Δ per month for 12 key biomarkers
- Volatility: std deviation across timeline for 4 key markers
- Reference range violations: count of high/low/critical values

**Explainability** — real SHAP TreeExplainer on every prediction. Feature attribution bars in the dashboard show which biomarkers drove the score.

---

## Claude AI integration

Three distinct use cases — each uses a different model and prompt strategy:

| Use case | Model | Input | Output |
|---|---|---|---|
| Lab report Vision OCR | claude-sonnet-4-20250514 | Base64 image | Structured JSON of biomarker values |
| Prediction narrative | claude-haiku-4-5-20251001 | ML prediction dict | 3–4 sentence plain-English summary |
| Trend anomaly detection | claude-sonnet-4-20250514 | Full biomarker timeline | Flagged patterns with clinical significance |
| Drug interaction explain | claude-haiku-4-5-20251001 | OpenFDA raw + med list | Plain-English interaction explanation |

The local ML models handle all risk scoring. Claude handles language tasks only (extraction, summarisation, narration). Never use an LLM for a clinical risk number.

---

## Data flow — prediction

```
Patient visits → Checkup added (30+ fields) → Cache invalidated
                         ↓
              BioSentinelEngine.predict()
              ├── _extract() — builds 49-feature vector from timeline
              ├── scaler.transform()
              ├── GBM.predict() × 4 domains
              ├── iso.predict() — calibrate to [0, 1]
              └── shap.TreeExplainer() — feature attribution
                         ↓
              DBPrediction saved → DBAlert generated → Email sent
                         ↓
              Claude Haiku → plain-English narrative (if API key set)
              Claude Sonnet → anomaly scan (on demand)
```

---

## Security model

| Layer | Implementation |
|---|---|
| Authentication | JWT HS256, 24h expiry, token revocation |
| 2FA | TOTP (pyotp) — Google Authenticator compatible |
| Multi-user isolation | Every patient row has `owner_id`; `_get_patient_or_403()` on every endpoint |
| Admin override | `role == "admin"` bypasses ownership for cross-patient access |
| Password storage | bcrypt via passlib |
| Field encryption | Fernet AES-256 (optional, HIPAA path) — set `FIELD_ENCRYPTION_KEY` |
| Rate limiting | slowapi 200/min global, 10/min on auth endpoints |
| Audit log | Every patient access, prediction, and config change logged immutably |
| CORS | `ALLOWED_ORIGINS` env var — default `*` (development), restrict in production |

---

## Deployment options

| Mode | Command | Best for |
|---|---|---|
| Local dev | `python run.py` | Development, testing |
| Docker + SQLite | `docker-compose up -d` | Single-server clinic |
| Docker + PostgreSQL | `docker-compose --profile postgres up -d` | Multi-user production |
| Docker + Nginx | `docker-compose --profile production up -d` | Public-facing deployment |
| PWA (installed) | Add to home screen via browser | Mobile clinician use |

---

## Caching

In-memory TTL cache (no Redis required) caches:
- Biomarker trend data (`GET /api/v1/patients/{pid}/trends`) — 5 min TTL
- Automatically invalidated when a new checkup is added for that patient
- `GET /api/v1/cache/stats` — live stats (admin/clinician only)
- `POST /api/v1/cache/flush` — clear all (admin only)

---

## Internationalisation

Dashboard supports 6 languages with runtime switching:

| Code | Language | Script | Direction |
|---|---|---|---|
| `en` | English | Latin | LTR |
| `hi` | Hindi | Devanagari | LTR |
| `es` | Spanish | Latin | LTR |
| `ta` | Tamil | Tamil script | LTR |
| `pt` | Portuguese | Latin | LTR |
| `ar` | Arabic | Arabic | **RTL** |

Arabic triggers automatic RTL layout (`dir="rtl"` on `<html>`).
Language preference persisted in `localStorage`.

---

## What's NOT in scope (by design)

- **No AI-generated risk scores** — LLMs are used for language tasks only; all clinical risk numbers come from scikit-learn models
- **No MIMIC data sent externally** — the PhysioNet DUA prohibits it; `train_mimic.py` runs entirely locally
- **No real-time streaming** — WebSockets not needed; polling every 30s is sufficient for clinical dashboards
- **No multi-tenancy** — each deployment is for one clinic; role-based isolation handles multi-user within a clinic
