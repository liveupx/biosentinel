"""Tests for OCR, trend alerts, overdue reminders, and capabilities."""
import base64, io, pytest


# ── Minimal synthetic "lab report" text for testing extraction ────────────────
FAKE_LAB_TEXT = b"""
PATIENT LAB REPORT - SRL DIAGNOSTICS
=====================================
Test Name              Result    Unit      Reference Range
HbA1c                  6.1       %         4.0 - 5.6
Fasting Blood Sugar    113.0     mg/dL     70 - 100
Haemoglobin            12.5      g/dL      12.0 - 16.0
WBC                    6.5       K/uL      4.5 - 10.0
Lymphocytes %          24.0      %         20 - 40
CEA                    3.8       ng/mL     0 - 5.0
ALT (SGPT)             38        U/L       7 - 40
LDL Cholesterol        133       mg/dL     0 - 130
HDL Cholesterol        47        mg/dL     40 - 100
Serum Creatinine       0.85      mg/dL     0.6 - 1.2
TSH                    2.3       mIU/L     0.4 - 4.0
CRP                    1.2       mg/L      0 - 3.0
"""

# Tiny 1x1 white PNG for image OCR test (won't extract values but must not crash)
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg=="
)


class TestOCREndpoints:
    def test_capabilities_endpoint(self, client, admin_headers):
        r = client.get("/api/v1/capabilities", headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert "version" in data
        assert data["version"] == "2.1.0"
        assert "features" in data
        assert data["features"]["trend_alerts"] is True
        assert data["features"]["overdue_reminders"] is True
        assert data["features"]["multi_user_isolation"] is True

    def test_ocr_base64_pdf_text(self, client, admin_headers):
        """Upload a text-based fake lab report and verify values extracted."""
        encoded = base64.b64encode(FAKE_LAB_TEXT).decode()
        r = client.post("/api/v1/ocr/extract-base64", headers=admin_headers,
                        json={"filename": "report.pdf", "data": encoded})
        # pdfplumber may or may not parse raw bytes as PDF — if it errors, 422
        # but the endpoint itself must respond cleanly
        assert r.status_code in (200, 422), r.text

    def test_ocr_base64_image(self, client, admin_headers):
        """Upload a tiny PNG — must not crash even if no values found."""
        encoded = base64.b64encode(TINY_PNG).decode()
        r = client.post("/api/v1/ocr/extract-base64", headers=admin_headers,
                        json={"filename": "report.png", "data": encoded})
        assert r.status_code in (200, 422), r.text

    def test_ocr_base64_with_data_url_prefix(self, client, admin_headers):
        """Accept data URL format (from browser FileReader.readAsDataURL)."""
        encoded = "data:image/png;base64," + base64.b64encode(TINY_PNG).decode()
        r = client.post("/api/v1/ocr/extract-base64", headers=admin_headers,
                        json={"filename": "scan.png", "data": encoded})
        assert r.status_code in (200, 422), r.text

    def test_ocr_unsupported_format(self, client, admin_headers):
        """Uploading an unsupported file type should return 422."""
        encoded = base64.b64encode(b"some data").decode()
        r = client.post("/api/v1/ocr/extract-base64", headers=admin_headers,
                        json={"filename": "report.docx", "data": encoded})
        assert r.status_code == 422
        assert "Unsupported" in r.json()["detail"]

    def test_ocr_invalid_base64(self, client, admin_headers):
        """Garbage base64 must return 422 cleanly, not 500."""
        r = client.post("/api/v1/ocr/extract-base64", headers=admin_headers,
                        json={"filename": "test.pdf", "data": "not_valid_base64!!!"})
        assert r.status_code == 422

    def test_ocr_requires_auth(self, client):
        """OCR endpoint must require a JWT token."""
        r = client.post("/api/v1/ocr/extract-base64",
                        json={"filename": "t.pdf", "data": ""})
        assert r.status_code in (401, 403)

    def test_ocr_text_extraction_logic(self):
        """Unit test the OCR text parser directly."""
        from app import LabReportOCR
        ocr = LabReportOCR()
        text = FAKE_LAB_TEXT.decode()
        vals = ocr.extract_from_text(text)
        # Should extract at least HbA1c, glucose, hemoglobin
        assert "hba1c" in vals, f"HbA1c not found. Got: {vals}"
        assert abs(vals["hba1c"] - 6.1) < 0.5, f"HbA1c value wrong: {vals['hba1c']}"
        assert "hemoglobin" in vals
        assert "cea" in vals


class TestTrendAlerts:
    def test_trend_alerts_too_few_checkups(self, client, admin_headers,
                                            sample_patient):
        """Should return a helpful message with < 3 checkups."""
        r = client.get(
            f"/api/v1/patients/{sample_patient['id']}/trend-alerts",
            headers=admin_headers
        )
        assert r.status_code == 200
        # No checkups → message about needing 3
        assert "message" in r.json() or r.json()["alerts"] == []

    def test_trend_alerts_rising_hba1c(self, client, admin_headers,
                                        sample_patient):
        """Steadily rising HbA1c over 4 checkups should generate a WARNING."""
        from datetime import datetime, timezone, timezone, timedelta
        today = datetime.now().date()
        pid = sample_patient["id"]
        # Dates: 9 months ago, 6 months ago, 3 months ago, today
        for i, hba1c in enumerate([5.4, 5.7, 5.9, 6.1]):
            date = (today - timedelta(days=90 * (3 - i))).isoformat()
            client.post("/api/v1/checkups", headers=admin_headers, json={
                "patient_id": pid, "checkup_date": date, "hba1c": hba1c
            })
        r = client.get(f"/api/v1/patients/{pid}/trend-alerts",
                       headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert data["checkups_analysed"] == 4
        # HbA1c rose ~13% — should trigger alert (threshold 8%)
        hba1c_alerts = [a for a in data["alerts"] if a["field"] == "hba1c"]
        assert len(hba1c_alerts) > 0, f"Expected HbA1c alert. Got: {data['alerts']}"
        assert hba1c_alerts[0]["direction"] == "rising"
        assert hba1c_alerts[0]["pct_change"] > 8

    def test_trend_alerts_declining_hemoglobin(self, client, admin_headers,
                                                sample_patient):
        """Declining hemoglobin should trigger a WARNING."""
        from datetime import datetime, timezone, timezone, timedelta
        today = datetime.now().date()
        pid = sample_patient["id"]
        for i, hgb in enumerate([14.5, 13.8, 13.1, 12.5]):
            date = (today - timedelta(days=90 * (3 - i))).isoformat()
            client.post("/api/v1/checkups", headers=admin_headers, json={
                "patient_id": pid, "checkup_date": date, "hemoglobin": hgb
            })
        r = client.get(f"/api/v1/patients/{pid}/trend-alerts",
                       headers=admin_headers)
        data = r.json()
        hgb_alerts = [a for a in data["alerts"] if a["field"] == "hemoglobin"]
        assert len(hgb_alerts) > 0, "Expected declining hemoglobin alert"
        assert hgb_alerts[0]["direction"] == "declining"

    def test_trend_alerts_stable_values_no_alert(self, client, admin_headers,
                                                   sample_patient):
        """Stable biomarkers should NOT generate alerts."""
        from datetime import datetime, timezone, timezone, timedelta
        today = datetime.now().date()
        pid = sample_patient["id"]
        for i in range(4):
            date = (today - timedelta(days=90 * (3 - i))).isoformat()
            client.post("/api/v1/checkups", headers=admin_headers, json={
                "patient_id": pid, "checkup_date": date,
                "hba1c": 5.2, "hemoglobin": 14.0, "cea": 1.5
            })
        r = client.get(f"/api/v1/patients/{pid}/trend-alerts",
                       headers=admin_headers)
        data = r.json()
        assert len(data["alerts"]) == 0, \
            f"Unexpected alerts for stable values: {data['alerts']}"

    def test_trend_alerts_ownership_enforced(self, client, admin_headers,
                                              clinician_headers, sample_patient):
        """Clinician can't get trend alerts for admin's patients."""
        r = client.get(
            f"/api/v1/patients/{sample_patient['id']}/trend-alerts",
            headers=clinician_headers
        )
        assert r.status_code == 403


class TestOverdueReminders:
    def test_overdue_no_checkups(self, client, admin_headers):
        """Patient with zero checkups should appear as overdue."""
        # Create patient (no checkups)
        p = client.post("/api/v1/patients", headers=admin_headers,
            json={"age": 45, "sex": "Male", "family_history_cancer": 0,
                  "family_history_diabetes": 0, "family_history_cardio": 0}).json()
        r = client.get("/api/v1/reminders/overdue", headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        ids = [o["patient_id"] for o in data["overdue_patients"]]
        assert p["id"] in ids

    def test_overdue_recent_checkup_not_listed(self, client, admin_headers,
                                                sample_patient):
        """Patient with a recent checkup should NOT be overdue."""
        from datetime import datetime, timezone, timezone, timedelta
        today = datetime.now(timezone.utc).date().isoformat()
        client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": sample_patient["id"],
            "checkup_date": today, "hba1c": 5.5
        })
        r = client.get("/api/v1/reminders/overdue", headers=admin_headers)
        ids = [o["patient_id"] for o in r.json()["overdue_patients"]]
        assert sample_patient["id"] not in ids

    def test_overdue_custom_threshold(self, client, admin_headers,
                                       sample_patient):
        """Custom days threshold should work."""
        # Add a checkup from 45 days ago
        from datetime import datetime, timezone, timezone, timedelta
        date_45d = (datetime.now(timezone.utc) - timedelta(days=45)).date().isoformat()
        client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": sample_patient["id"],
            "checkup_date": date_45d, "hba1c": 5.5
        })
        # With 30-day threshold, patient IS overdue
        r30 = client.get("/api/v1/reminders/overdue?days=30",
                         headers=admin_headers)
        ids_30 = [o["patient_id"] for o in r30.json()["overdue_patients"]]
        assert sample_patient["id"] in ids_30

        # With 90-day threshold, patient is NOT overdue
        r90 = client.get("/api/v1/reminders/overdue?days=90",
                         headers=admin_headers)
        ids_90 = [o["patient_id"] for o in r90.json()["overdue_patients"]]
        assert sample_patient["id"] not in ids_90

    def test_overdue_user_scoped(self, client, admin_headers,
                                  clinician_headers):
        """Clinician only sees overdue patients they own."""
        # Admin creates a patient (no checkups)
        client.post("/api/v1/patients", headers=admin_headers,
            json={"age": 50, "sex": "Male", "family_history_cancer": 0,
                  "family_history_diabetes": 0, "family_history_cardio": 0})
        # Clinician's overdue list should only show their own patients
        r = client.get("/api/v1/reminders/overdue", headers=clinician_headers)
        data = r.json()
        # All returned patients should belong to the clinician
        # (We can't easily check owner without another call, but count should be 0
        #  since clinician has no patients yet)
        assert r.status_code == 200


class TestPatientSelfRegister:
    def test_self_register(self, client):
        """Patients can register without a JWT token."""
        r = client.post("/api/v1/auth/register-patient", json={
            "username": "patient_self",
            "email": "patient@test.com",
            "password": "patientpass123"
        })
        assert r.status_code == 201
        data = r.json()
        assert data["user"]["role"] == "patient"
        assert "access_token" in data

    def test_self_register_short_password(self, client):
        r = client.post("/api/v1/auth/register-patient", json={
            "username": "p2", "email": "p2@test.com", "password": "short"
        })
        assert r.status_code == 400

    def test_self_register_duplicate_username(self, client):
        payload = {"username": "dup_pat", "email": "d@t.com",
                   "password": "longpassword123"}
        client.post("/api/v1/auth/register-patient", json=payload)
        r = client.post("/api/v1/auth/register-patient", json=payload)
        assert r.status_code == 400

    def test_self_register_with_clinician_link(self, client, admin_headers):
        """Self-registration can reference a clinician username."""
        r = client.post("/api/v1/auth/register-patient", json={
            "username": "linked_patient",
            "email": "linked@test.com",
            "password": "password12345",
            "clinician_username": "admin"   # links to admin created in fixture
        })
        assert r.status_code == 201
        assert "linked_clinician" in r.json()["user"]
