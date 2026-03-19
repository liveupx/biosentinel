# Changelog

All notable changes to BioSentinel are documented here.
Format: [Keep a Changelog](https://keepachangelog.com) · [Semantic Versioning](https://semver.org)

---

## [2.3.4] — 2026-03-19 (Truly Final)

### Fixed
- `scheduler.py` — 3 `datetime.utcnow()` calls replaced with `datetime.now(timezone.utc)`. All deprecation warnings now eliminated across the entire codebase.
- All remaining `v2.1.0`/`v2.1` version strings updated to `v2.3.3`/`v2.3` across `pyproject.toml`, `app.py`, `START_MAC_LINUX.sh`, `START_WINDOWS.bat`, `biosentinel_patient_portal.html`, `OPEN_COLLECTIVE_PITCH.md`, `tests/test_ocr.py`.
- API URL hardcoded to `http://localhost:8000` in all 3 HTML files — replaced with dynamic detection: uses `window.location.origin` in production, falls back to `localhost:8000` when running locally.

### Added
- **`biosentinel_sdk.py`** — typed Python SDK client for the BioSentinel API. Wraps all major endpoints: patients, checkups, predictions, SHAP, AI narratives, anomaly detection, percentiles, alerts, medications, analytics. Drop-in for external integrations. `pip install httpx` is the only dependency.
- **`tests/test_sdk.py`** — 10 SDK tests covering init, error handling, connect(), health check.
- **`.gitignore`** — comprehensive gitignore covering Python, venvs, DB files, SSL certs, MIMIC data (with explicit DUA reminder comment), MLflow runs, logs.
- **`CODE_OF_CONDUCT.md`** — healthcare-specific CoC including medical ethics section for clinical software.
- **`CITATION.cff`** — Citation File Format for academic use, with literature references.

### Final Counts
- **273 tests** passing — zero failures, zero deprecation warnings from our code
- **7 Python modules** (app, claude_ai, scheduler, mlflow_tracking, train_mimic, migrate_to_postgres, biosentinel_sdk)
- **3 HTML interfaces** (dashboard, patient portal, patient view)  
- **4 infrastructure files** (Dockerfile, docker-compose, nginx.conf, sw.js)
- **6 languages** in dashboard + portal
- **42+ API endpoints**
- **0 TODO/FIXME** left in source

---

## [2.3.3] — 2026-03-19 (Absolute Final)

### Fixed & Completed
- **`03_biomarker_trend_analysis.ipynb`** — fully rewritten to be self-contained and runnable from the repo root. No broken `../data/samples/` paths. Handles the nested `lab_results` JSON structure (cbc/metabolic_panel/etc.). All 5 code cells execute cleanly producing 3 charts saved to `img/`.
- **`tests/test_2fa_fhir.py`** — `pyotp` added to CI install step; all 22 TOTP + FHIR tests now pass.
- **`app.py`** — final 2 `body.dict(exclude_none=True)` → `body.model_dump(exclude_none=True)` in notification preferences endpoint. Zero Pydantic warnings.

### Added
- **`img/biomarker_trajectories.png`** — 6-panel longitudinal chart: HbA1c, Glucose, Hemoglobin, Lymphocytes, CEA, LDL plotted with reference range bands and slope annotations. Generated from patient_sample.json.
- **`img/shap_attribution.png`** — SHAP feature attribution bar chart showing risk drivers (red = increases risk, green = protective). Generated programmatically.
- **`img/risk_trajectory.png`** — risk score evolution line chart across all 4 domains over time.
- README screenshots section updated with the 3 new charts.

### Final Counts
- **263 tests** passing across all 12 test files — **zero failures, zero Pydantic warnings**
- **6 Python modules**: app.py, claude_ai.py, scheduler.py, mlflow_tracking.py, train_mimic.py, migrate_to_postgres.py
- **3 HTML interfaces**: biosentinel_dashboard.html, biosentinel_patient_portal.html, biosentinel_patient_view.html
- **PWA**: sw.js, manifest.json, offline.html
- **6 languages**: English, Hindi, Spanish, Tamil, Portuguese, Arabic (RTL)
- **42 endpoints**: 37 original + 5 new AI/cache/percentile endpoints

---

## [2.3.2] — 2026-03-19 (Final)

### Fixed
- `tests/test_new_features.py` — `TestingSession` import used wrong path (`tests.conftest` → `conftest`); test now uses `try/finally` for safe session cleanup. Both password-reset flow tests now pass.
- `_fhir_age()` — `datetime.fromisoformat()` returned a naive datetime while `datetime.now(timezone.utc)` returned an aware one; subtraction raised `TypeError`. Fixed by using `date.fromisoformat()` and `date.today()` (timezone-agnostic).
- `app.py` — final 2 `body.dict(exclude_none=True)` calls in notification preferences endpoint replaced with `body.model_dump(exclude_none=True)`. Zero Pydantic deprecation warnings remain in app.py.

### Updated
- `.github/workflows/ci.yml` — removed test file ignores; all 263 tests now run in CI. Added `pyotp qrcode[pil]` install step for 2FA tests.
- `README.md` — test badge updated to 263 passing.

### Summary: Complete test suite
All **263 tests passing** across 12 test files:
- `test_auth.py` — JWT, register, login, password change
- `test_patients.py` — CRUD, multi-user isolation, search
- `test_checkups.py` — biomarker ingestion, trend direction
- `test_predictions.py` — ML predictions, alerts, SHAP
- `test_medications.py` — medications, diagnoses, diet plans
- `test_analytics.py` — population stats, audit log, reports
- `test_ocr.py` — OCR endpoints, overdue reminders, capabilities
- `test_claude_ai.py` — Claude AI endpoints (monkeypatched)
- `test_percentile.py` — percentile comparison endpoints
- `test_2fa_fhir.py` — TOTP 2FA, FHIR import, age calculation
- `test_complete_features.py` — encryption, webhooks, batch predictions, search, genomics
- `test_new_features.py` — SHAP, password reset, bulk import, PostgreSQL, notifications

---

## [2.3.1] — 2026-03-19

### Added — Final completions
- **nginx.conf** fully rewritten — HTTPS/HTTP2 with TLS 1.2/1.3 and strong ciphers, HSTS, CSP header, per-zone rate limiting (API 60/min, auth 10/min, OCR 10/min), PWA-specific cache headers (`sw.js` no-cache, `manifest.json` 24h, static assets 30d), proxy keep-alive, 25MB body limit for OCR uploads, blocks `.env`/`.git` paths.
- **Patient portal i18n** — English, Hindi, Spanish, Tamil, Portuguese with language switcher. Auto-detects browser language on first load.
- **PWA to all pages** — `manifest.json` + `theme-color` meta + SW registration added to `biosentinel_patient_portal.html` and `index.html`.
- **GitHub issue templates** — `bug_report.md`, `feature_request.md`, `clinical_validation.md`, `config.yml` (with contact links for security, clinical partnerships, Open Collective).
- **`SCREENSHOTS.html`** — visual feature overview covering all v2.1–v2.3 additions with interactive mock-ups.
- **Cache SET fixed** — the `_cache_set` call was missing from the trends endpoint return path; fixed.
- **`CONTRIBUTING.md`** updated with v2.3 file map and quick-start commands.
- **`run.py`** updated to list patient portal URL in startup messages.
- **`OPEN_COLLECTIVE_PITCH.md`** — complete sponsorship pitch with problem statement, what's built, who it's for, funding goals, and honest limitations section.

---

## [2.3.0] — 2026-03-19

### Added — Features
- **`train_mimic.py`** — complete MIMIC-IV training script with BigQuery export SQL, feature engineering matching app.py exactly, outcome labelling from ICD codes, AUC/MAE evaluation, and `--compare` mode for synthetic vs real data comparison.
- **PWA support** — `sw.js` (service worker: cache-first static, network-first API, background sync, push notifications), `manifest.json` (installable on iOS/Android), `offline.html` (auto-retry every 10s), PWA meta tags + SW registration in dashboard.
- **In-memory TTL cache** — no Redis required; `_cache_get/set/invalidate()` with `threading.Lock`. Caches biomarker trend responses for 5 min; auto-invalidated on new checkup. `GET /api/v1/cache/stats` and `POST /api/v1/cache/flush` endpoints added.
- **`architecture.md`** — fully rewritten to reflect v2.2 accurately: system diagram, file table, ML pipeline, Claude AI use cases, security model, deployment matrix, i18n table, caching.

### Updated
- `biosentinel_dashboard.html` — Arabic RTL layout fix (`dir="rtl"` + `lang` attribute); language switcher initialised on login; PWA install prompt button wiring.
- `requirements.txt` — added `mlflow>=2.12.0`.
- `CACHE_TTL_SECONDS` env var added to `.env.example`.

---

## [2.2.0] — 2026-03-19

### Added — Features
- **Percentile comparison** (`GET /api/v1/analytics/percentile/{pid}`) — compares a patient's latest risk scores against all patients in the same age band (±10 years) and sex. Returns percentile rank per domain (0–100) with plain-English interpretation.
- **Biomarker percentile** (`GET /api/v1/analytics/biomarker-percentile/{pid}?biomarker=hba1c`) — compares a specific lab value against the comparable population. Shows group avg, median, min, max.
- **Multi-language dashboard** — added Spanish (es), Tamil (ta), Portuguese (pt), Arabic (ar) to the existing English + Hindi i18n system. Arabic triggers RTL layout automatically. Language switcher buttons appear in the header bar after login.
- **`mlflow_tracking.py`** — optional MLflow integration for experiment tracking. Set `MLFLOW_TRACKING=1` to log every training run: hyperparameters, MAE per domain, top-10 feature importances, model artefacts. Run `mlflow ui --port 5001` to compare runs visually. No-op when disabled.
- **`OPEN_COLLECTIVE_PITCH.md`** — structured pitch document for Open Collective + healthcare community outreach.

### Added — Tests
- **`tests/test_percentile.py`** — 13 tests covering both percentile endpoints: no prediction, too few patients, full percentile with 5+ comparable patients, auth guards, cross-patient isolation.

### Updated
- `requirements.txt` — added `mlflow>=2.12.0` (optional).
- OCR patterns hardened: `hba1c` regex catches OCR artefacts (`HbAIc`, `Hb A1 C`); `ca125` and `lymphocytes_pct` patterns broadened for real-world lab formats.
- `app.py` — `MLFLOW_AVAILABLE` flag + MLflow status included in `/api/v1/system-info`.

---

## [2.1.1] — 2026-03-19

### Fixed
- `datetime.utcnow()` → `datetime.now(timezone.utc)` — **41 occurrences** across app.py and all test files. Eliminates all Python 3.12 deprecation warnings from CI logs.
- Pydantic v2: `.dict()` → `.model_dump()` — 6 occurrences fixed (patient update, checkup add, medication/diagnosis/diet plan add).
- `tests/test_ocr.py` hardcoded version string updated to `2.1.0`.
- `urllib.parse` import missing in medication interaction endpoint — added alongside `urllib.request`.

### Added
- `migrate_to_postgres.py` — one-command SQLite → PostgreSQL migration with batch copy, progress output, and `--verify` mode to confirm row counts match.
- `sample_lab_report.jpg` — synthetic lab report for OCR pipeline validation (PathCare format, 18 biomarkers).
- OCR pattern improvements: `hba1c` regex now matches OCR artefacts (`HbAIc`, `Hb A1 C`); `ca125` and `lymphocytes_pct` patterns broadened.

### Updated
- `pyproject.toml` version: `2.0.0` → `2.1.0`
- `.github/FUNDING.yml` — `open_collective`, `ko_fi`, `buy_me_a_coffee` fields populated.
- `docker-compose.yml` — PostgreSQL service added (`--profile postgres`), `ANTHROPIC_API_KEY` wired, `biosentinel_patient_portal.html` served by Nginx, healthcheck start_period 60s → 90s.
- `Dockerfile` — `tesseract-ocr` and `poppler-utils` installed in runtime image; `claude_ai.py` and `scheduler.py` copied; OCI labels added.
- `.github/workflows/ci.yml` — deprecation guard steps added (CI fails if `utcnow()` or `.dict()` re-appears); lints `claude_ai.py` and `scheduler.py` too.
- `README.md` — feature table, project structure, URL table, Known Limitations updated for v2.1.

---

## [2.1.0] — 2026-03-19

### Added — Claude AI Integration
- **`claude_ai.py`** — new module with 5 Claude-powered capabilities:
  - `extract_labs_from_image()` — Claude Sonnet vision reads any lab report photo/scan and returns structured biomarker JSON
  - `extract_labs_from_pdf_pages()` — PDF→image→Vision pipeline via `pdf2image` + Claude
  - `generate_prediction_narrative()` — Claude Haiku writes plain-English patient/clinician summaries after every prediction
  - `detect_trend_anomalies()` — Claude Sonnet analyses the full longitudinal record for subtle patterns that fixed thresholds miss
  - `explain_drug_interactions()` — Claude Haiku converts raw OpenFDA interaction data into plain-English explanations
- **`/api/v1/ocr/claude-vision`** — new endpoint: upload any lab image/PDF, Claude reads it, auto-fills checkup form
- **`/api/v1/ai/narrative/{pid}`** — generate patient or clinician narrative for latest prediction
- **`/api/v1/ai/anomalies/{pid}`** — longitudinal trend anomaly detection via Claude Sonnet
- **`/api/v1/ai/status`** — returns Claude AI integration status and feature availability
- **`/api/v1/medications/{mid}/interaction-explain`** — plain-English drug interaction explanation via OpenFDA + Claude Haiku

### Added — Background Scheduler
- **`scheduler.py`** — APScheduler daemon (zero config, no Redis/Celery required):
  - `overdue_checkup_scan` — runs at 08:00 UTC daily, finds patients >90 days since last checkup, emails their clinician
  - `daily_stats_log` — logs patient/alert/overdue counts for monitoring at 06:00 UTC
  - Gracefully degrades if `apscheduler` not installed

### Added — Dashboard AI Features
- New **✦ AI Insights** tab on every patient profile with three panels:
  - **AI Narrative** — one-click patient or clinician summary generation
  - **Longitudinal Anomaly Detection** — full timeline scan with flagged patterns
  - **Claude Vision OCR** — drag-and-drop lab report upload with auto-fill
- AI Engine Status card showing live feature availability

### Fixed — Repo Structure
- Moved all 10 test files + `conftest.py` into `tests/` directory (CI now actually runs)
- Moved `ci.yml` → `.github/workflows/ci.yml` (GitHub Actions was never triggered before)
- Deleted orphaned scaffolding: `main.py`, `base_model.py`, `cancer_risk_net.py` (all imported `src/` package that was never built)

### Added — Configuration
- `ANTHROPIC_API_KEY` env var — enables all Claude AI features
- `OVERDUE_REMINDER_DAYS` env var — configures reminder threshold (default: 90)
- `.env.example` rewritten with complete documentation of all env vars
- `requirements.txt` updated with: `anthropic`, `apscheduler`, `pdf2image`, `pytest-cov`

### Tests
- **`tests/test_claude_ai.py`** — 27 new tests covering all AI endpoints with monkeypatched stubs (no real API calls in CI)
- Total: **69 passing tests** across auth, patients, checkups, predictions, medications, analytics, AI

---

## [2.0.0] — 2025-03-17

### Added — Backend
- **Multi-user data isolation**: each clinician only sees their own patients; admin sees all
- `owner_id` field on every Patient record — set automatically at creation
- `_get_patient_or_403()` helper — 403 HTTP error if unauthorized access attempted
- `DBEmailConfig` model — per-user SMTP configuration stored securely
- Full SMTP email alert engine — HTML-formatted alert emails on CRITICAL/HIGH predictions
- `GET/PUT /api/v1/settings/email` — save SMTP config; `POST /settings/email/test` — test connection
- `PUT /api/v1/auth/password` — password change endpoint with validation
- `DBAuditLog` model — immutable audit trail of every patient data access
- `GET /api/v1/audit-log` — viewable by admins (all) and users (own actions)
- Overdue checkup detection — patients with 90+ day gap flagged in `/stats`
- User-scoped analytics — each user's population stats reflect only their patients
- `DBDietPlan` model and `POST /api/v1/diet-plans`, `GET /api/v1/patients/{id}/diet-plans`
- `GET /api/v1/analytics/risk-trajectory/{pid}` — multi-point risk score history per patient
- All 37 API endpoints now enforce ownership correctly

### Added — Frontend (clinician dashboard)
- Settings page: email SMTP config form, enable/disable toggle, send test button
- Settings page: password change form with confirmation
- Risk trajectory chart — line chart showing score evolution across multiple predictions
- Audit log viewer — full access history table
- Overdue checkup banner on dashboard home
- Diet Plans tab in patient profile with add form
- PDF report now opens with full patient data including risk scores, meds, diagnoses
- Patient list shows overdue badge for patients needing checkups

### Added — Testing
- 72 real pytest tests (all passing) covering:
  - Authentication: register, login, token validation, password change
  - Multi-user isolation: 7 dedicated security tests
  - Patient CRUD: create, read, update, delete, cascade
  - Checkup ingestion: 30+ biomarker fields, ordering, count
  - Biomarker trends: direction detection, status flags, reference ranges
  - AI predictions: calibration, domains, top features, persistence
  - Clinical alerts: generation, acknowledgement, unread counts
  - Risk trajectory: multi-prediction accumulation
  - Analytics: population stats, user scoping
  - Reports: structure, disclaimer presence
  - Email settings: save + masked password response
  - Audit log: user scoping, action recording

### Added — Infrastructure
- `Dockerfile` — multi-stage production build, non-root user, health check
- `docker-compose.yml` — API service + optional Nginx production profile
- `nginx.conf` — reverse proxy with rate limiting, security headers
- `requirements.txt` — pinned dependencies for reproducibility
- `.env.example` — full configuration template with Gmail/Outlook/SMTP setup guides
- `pytest.ini` — test configuration

### Fixed
- Multi-user isolation was completely absent in v1.0 — all users could see all patients
- Email config PUT endpoint now properly persists all fields
- SQLite DB path now configurable via `DATABASE_URL` env var (required for testing)
- `get_predictions` endpoint now enforces patient ownership

---

## [1.0.0] — 2025-03-16

### Added
- 4 calibrated ML models: cancer, metabolic, cardio, hematologic (GBM + isotonic calibration)
- Full REST API: 30+ endpoints for patients, checkups, medications, diagnoses, predictions
- JWT authentication with roles (admin/clinician/researcher)
- SQLite database with 8 entity models
- Clinician dashboard (dark theme, technical)
- Non-technical patient view (plain English, color-coded risk lights)
- One-click launcher (`run.py`, `START_WINDOWS.bat`, `START_MAC_LINUX.sh`)
- 5 seeded demo patients covering full risk spectrum
- Clinical alert system (CRITICAL/WARNING/INFO)
- Biomarker trend analysis with Chart.js visualizations
- Population analytics with risk distributions
- Patient health report (HTML/PDF)
- Overdue patient detection
