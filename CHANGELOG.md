# Changelog

All notable changes to BioSentinel are documented here.
Format: [Keep a Changelog](https://keepachangelog.com) · [Semantic Versioning](https://semver.org)

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
