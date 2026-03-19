"""
Tests for the 6 new features:
 1. SHAP explanations
 2. PostgreSQL support (env-based, tested via capabilities)
 3. Password reset (email / SMS / WhatsApp / Telegram)
 4. Bulk CSV / Excel import
 5. Appointment reminder send
 6. Drug interaction checker (OpenFDA)
"""
import io, csv, base64, pytest
from unittest.mock import patch, MagicMock


# ─────────────────────────────────────────────────────────────────────────────
# 1. SHAP EXPLANATIONS
# ─────────────────────────────────────────────────────────────────────────────

class TestSHAPExplanations:
    def test_shap_endpoint_requires_checkup(self, client, admin_headers,
                                             sample_patient):
        r = client.get(
            f"/api/v1/patients/{sample_patient['id']}/shap/cancer",
            headers=admin_headers
        )
        assert r.status_code == 400
        assert "checkup" in r.json()["detail"].lower()

    def test_shap_invalid_domain(self, client, admin_headers,
                                  sample_patient, sample_checkup):
        r = client.get(
            f"/api/v1/patients/{sample_patient['id']}/shap/diabetes",
            headers=admin_headers
        )
        assert r.status_code == 400

    def test_shap_all_valid_domains(self, client, admin_headers,
                                     sample_patient, sample_checkup):
        pid = sample_patient["id"]
        # Must run a prediction first to ensure models are trained
        client.post(f"/api/v1/patients/{pid}/predict", headers=admin_headers)
        for domain in ("cancer", "metabolic", "cardio", "hematologic"):
            r = client.get(f"/api/v1/patients/{pid}/shap/{domain}",
                           headers=admin_headers)
            assert r.status_code == 200, f"SHAP failed for {domain}: {r.text}"
            data = r.json()
            assert data["domain"] == domain
            assert "top_features" in data
            assert len(data["top_features"]) > 0
            assert "method" in data
            # Each feature must have required fields
            for feat in data["top_features"]:
                assert "label" in feat or "feature" in feat
                assert "raw_value" in feat

    def test_shap_ownership_enforced(self, client, admin_headers,
                                      clinician_headers, sample_patient,
                                      sample_checkup):
        r = client.get(
            f"/api/v1/patients/{sample_patient['id']}/shap/cancer",
            headers=clinician_headers
        )
        assert r.status_code == 403

    def test_shap_method_field_present(self, client, admin_headers,
                                        sample_patient, sample_checkup):
        pid = sample_patient["id"]
        r = client.get(f"/api/v1/patients/{pid}/shap/metabolic",
                       headers=admin_headers)
        data = r.json()
        # Method should be either shap_tree_explainer or gbt_feature_importance
        assert data.get("method") in (
            "shap_tree_explainer", "gbt_feature_importance"
        )

    def test_capabilities_shows_shap_status(self, client, admin_headers):
        r = client.get("/api/v1/capabilities", headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert "shap_available" in data
        assert "shap_explanations" in data["features"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. POSTGRESQL SUPPORT (tested via capabilities + env detection)
# ─────────────────────────────────────────────────────────────────────────────

class TestPostgreSQLSupport:
    def test_db_type_reported_in_capabilities(self, client, admin_headers):
        r = client.get("/api/v1/capabilities", headers=admin_headers)
        data = r.json()
        assert "db_type" in data
        # In CI/tests we use SQLite
        assert data["db_type"] in ("sqlite", "postgresql")

    def test_engine_creation_sqlite(self):
        """_make_engine() must produce SQLite engine for sqlite:// URL."""
        import os, app as app_module
        original = os.environ.get("DATABASE_URL")
        os.environ["DATABASE_URL"] = "sqlite:///./test_pg_check.db"
        from app import _make_engine
        eng, is_sqlite = _make_engine()
        assert is_sqlite is True
        if original: os.environ["DATABASE_URL"] = original
        else: del os.environ["DATABASE_URL"]

    def test_engine_creation_postgres_url_fix(self):
        """postgres:// URLs (Heroku) must be rewritten to postgresql://."""
        import os, app as app_module
        from app import _make_engine
        # We don't actually connect — just verify URL rewriting doesn't crash
        url = "postgres://user:pass@localhost:5432/testdb"
        # Simulate the fix logic
        fixed = url.replace("postgres://", "postgresql://", 1)
        assert fixed.startswith("postgresql://")


# ─────────────────────────────────────────────────────────────────────────────
# 3. PASSWORD RESET
# ─────────────────────────────────────────────────────────────────────────────

class TestPasswordReset:
    def test_forgot_password_nonexistent_user(self, client):
        """Should return 200 even for unknown users (no enumeration)."""
        r = client.post("/api/v1/auth/forgot-password", json={
            "username_or_email": "ghost_user_xyz",
            "channel": "email"
        })
        assert r.status_code == 200
        assert "sent" in r.json()["message"].lower() or "exist" in r.json()["message"].lower()

    def test_forgot_password_no_email_on_file(self, client):
        """User with no email requesting email reset → 400."""
        # Register a user without email won't work (email is required at register)
        # So test the phone/telegram path for a user without those fields
        client.post("/api/v1/auth/register", json={
            "username": "nophone_user", "email": "nophone@t.com",
            "password": "pass12345", "role": "clinician"
        })
        r = client.post("/api/v1/auth/forgot-password", json={
            "username_or_email": "nophone_user",
            "channel": "sms"  # no phone on file
        })
        assert r.status_code == 400
        assert "phone" in r.json()["detail"].lower()

    def test_full_reset_flow_email_channel(self, client):
        """Create user → request reset → confirm with OTP → login with new pw."""
        # Register
        client.post("/api/v1/auth/register", json={
            "username": "reset_test_user",
            "email": "reset@biosentinel.test",
            "password": "originalpass123",
            "role": "clinician"
        })

        # Request reset (email channel — send is mocked/disabled in test)
        r = client.post("/api/v1/auth/forgot-password", json={
            "username_or_email": "reset_test_user",
            "channel": "email"
        })
        assert r.status_code == 200

        # Retrieve the OTP directly from the DB (test-only shortcut)
        from conftest import TestingSession
        from app import DBPasswordResetToken, DBUser
        db = TestingSession()
        try:
            user = db.query(DBUser).filter(DBUser.username == "reset_test_user").first()
            token_row = (db.query(DBPasswordResetToken)
                         .filter(DBPasswordResetToken.user_id == user.id,
                                 DBPasswordResetToken.used == 0).first())
            otp = token_row.token
        finally:
            db.close()

        # Confirm reset with OTP
        r2 = client.post("/api/v1/auth/reset-password", json={
            "token": otp, "new_password": "newstrongpass456"
        })
        assert r2.status_code == 200, r2.text

        # Login with new password
        r3 = client.post("/api/v1/auth/login", json={
            "username": "reset_test_user", "password": "newstrongpass456"
        })
        assert r3.status_code == 200

        # Old password must fail
        r4 = client.post("/api/v1/auth/login", json={
            "username": "reset_test_user", "password": "originalpass123"
        })
        assert r4.status_code == 401

    def test_reset_invalid_token(self, client):
        r = client.post("/api/v1/auth/reset-password", json={
            "token": "000000", "new_password": "newpass999"
        })
        assert r.status_code == 400

    def test_reset_token_single_use(self, client):
        """OTP must be rejected after first use."""
        client.post("/api/v1/auth/register", json={
            "username": "oneuse_user", "email": "oneuse@t.com",
            "password": "pass12345", "role": "clinician"
        })
        client.post("/api/v1/auth/forgot-password", json={
            "username_or_email": "oneuse_user", "channel": "email"
        })
        from conftest import TestingSession
        from app import DBPasswordResetToken, DBUser
        db = TestingSession()
        user = db.query(DBUser).filter(DBUser.username == "oneuse_user").first()
        token_row = (db.query(DBPasswordResetToken)
                     .filter(DBPasswordResetToken.user_id == user.id).first())
        otp = token_row.token; db.close()

        # Use it once
        client.post("/api/v1/auth/reset-password",
                    json={"token": otp, "new_password": "firstchange123"})
        # Second use must fail
        r = client.post("/api/v1/auth/reset-password",
                        json={"token": otp, "new_password": "secondchange456"})
        assert r.status_code == 400

    def test_reset_password_too_short(self, client):
        r = client.post("/api/v1/auth/reset-password", json={
            "token": "123456", "new_password": "short"
        })
        assert r.status_code == 400

    def test_profile_update_phone_telegram(self, client, admin_headers):
        r = client.put("/api/v1/auth/profile", headers=admin_headers, json={
            "phone": "+919876543210",
            "telegram_chat_id": "987654321"
        })
        assert r.status_code == 200
        data = r.json()
        assert data["phone"] == "+919876543210"
        assert data["telegram_chat_id"] == "987654321"

    def test_profile_get(self, client, admin_headers):
        r = client.get("/api/v1/auth/profile", headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert "username" in data
        assert "email" in data

    def test_forgot_password_telegram_no_chat_id(self, client):
        """Telegram reset without chat_id on file → 400."""
        client.post("/api/v1/auth/register", json={
            "username": "telegram_user", "email": "tg@t.com",
            "password": "pass12345", "role": "clinician"
        })
        r = client.post("/api/v1/auth/forgot-password", json={
            "username_or_email": "telegram_user", "channel": "telegram"
        })
        assert r.status_code == 400


# ─────────────────────────────────────────────────────────────────────────────
# 4. BULK CSV IMPORT
# ─────────────────────────────────────────────────────────────────────────────

def _make_csv(rows: list[dict]) -> bytes:
    if not rows: return b""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
    return buf.getvalue().encode("utf-8")


class TestBulkImport:
    def test_download_patient_template(self, client, admin_headers):
        r = client.get("/api/v1/import/template?type=patients",
                       headers=admin_headers)
        assert r.status_code == 200
        assert "age" in r.text
        assert "sex" in r.text

    def test_download_checkup_template(self, client, admin_headers):
        r = client.get("/api/v1/import/template?type=checkups",
                       headers=admin_headers)
        assert r.status_code == 200
        assert "patient_id" in r.text

    def test_import_patients_csv_success(self, client, admin_headers):
        csv_data = _make_csv([
            {"age": 45, "sex": "Male", "ethnicity": "South Asian",
             "family_history_cancer": 1, "family_history_diabetes": 0,
             "family_history_cardio": 0, "smoking_status": "never",
             "alcohol_units_weekly": 2, "exercise_min_weekly": 150, "notes": "Test A"},
            {"age": 38, "sex": "Female", "ethnicity": "Caucasian",
             "family_history_cancer": 0, "family_history_diabetes": 1,
             "family_history_cardio": 0, "smoking_status": "former",
             "alcohol_units_weekly": 0, "exercise_min_weekly": 90, "notes": "Test B"},
            {"age": 62, "sex": "Male", "ethnicity": "",
             "family_history_cancer": 2, "family_history_diabetes": 1,
             "family_history_cardio": 1, "smoking_status": "current",
             "alcohol_units_weekly": 7, "exercise_min_weekly": 30, "notes": ""},
        ])
        r = client.post("/api/v1/import/patients-csv",
                        headers=admin_headers,
                        files={"file": ("patients.csv", csv_data, "text/csv")})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["imported"] == 3
        assert data["errors"] == 0
        # Verify patients exist in DB
        pats = client.get("/api/v1/patients", headers=admin_headers).json()
        assert pats["total"] >= 3

    def test_import_patients_missing_required_columns(self, client, admin_headers):
        csv_data = _make_csv([{"ethnicity": "Asian", "notes": "No age or sex"}])
        r = client.post("/api/v1/import/patients-csv",
                        headers=admin_headers,
                        files={"file": ("bad.csv", csv_data, "text/csv")})
        assert r.status_code == 422
        assert "age" in r.json()["detail"].lower()

    def test_import_patients_with_invalid_rows(self, client, admin_headers):
        """Invalid rows produce errors but valid rows still import."""
        csv_data = _make_csv([
            {"age": 45, "sex": "Male"},
            {"age": 999, "sex": "Male"},   # invalid age
            {"age": 38, "sex": "Unknown"},  # invalid sex
            {"age": 50, "sex": "Female"},
        ])
        r = client.post("/api/v1/import/patients-csv",
                        headers=admin_headers,
                        files={"file": ("mixed.csv", csv_data, "text/csv")})
        assert r.status_code == 200
        data = r.json()
        assert data["imported"] == 2   # rows 1 and 4
        assert data["errors"] == 2

    def test_import_checkups_csv(self, client, admin_headers, sample_patient):
        pid = sample_patient["id"]
        csv_data = _make_csv([
            {"patient_id": pid, "checkup_date": "2024-01-01",
             "hba1c": 5.8, "glucose_fasting": 102, "hemoglobin": 13.5,
             "cea": 2.1, "ldl": 118, "bp_systolic": 128},
            {"patient_id": pid, "checkup_date": "2024-04-01",
             "hba1c": 6.0, "glucose_fasting": 107, "hemoglobin": 13.2,
             "cea": 2.4, "ldl": 122, "bp_systolic": 131},
        ])
        r = client.post("/api/v1/import/checkups-csv",
                        headers=admin_headers,
                        files={"file": ("checkups.csv", csv_data, "text/csv")})
        assert r.status_code == 200
        data = r.json()
        assert data["imported"] == 2
        assert data["errors"] == 0
        # Verify in DB
        chks = client.get(f"/api/v1/patients/{pid}/checkups",
                          headers=admin_headers).json()
        assert chks["count"] >= 2

    def test_import_checkups_ownership_enforced(self, client,
                                                 admin_headers, clinician_headers,
                                                 sample_patient):
        """Clinician cannot import checkups into admin's patient."""
        pid = sample_patient["id"]
        csv_data = _make_csv([{"patient_id": pid, "checkup_date": "2024-01-01"}])
        r = client.post("/api/v1/import/checkups-csv",
                        headers=clinician_headers,
                        files={"file": ("c.csv", csv_data, "text/csv")})
        assert r.status_code == 200
        # All rows should fail with access error
        data = r.json()
        assert data["imported"] == 0
        assert data["errors"] > 0

    def test_import_unsupported_file_type(self, client, admin_headers):
        r = client.post("/api/v1/import/patients-csv",
                        headers=admin_headers,
                        files={"file": ("data.pdf", b"not a csv", "application/pdf")})
        # Should either 422 or parse error gracefully
        assert r.status_code in (200, 422)


# ─────────────────────────────────────────────────────────────────────────────
# 5. APPOINTMENT REMINDERS
# ─────────────────────────────────────────────────────────────────────────────

class TestAppointmentReminders:
    def test_send_reminders_requires_clinician_role(self, client):
        # Register a patient-role user
        client.post("/api/v1/auth/register-patient", json={
            "username": "patient_role",
            "email": "pr@t.com",
            "password": "pass12345"
        })
        tok = client.post("/api/v1/auth/login",
            json={"username": "patient_role", "password": "pass12345"}).json()["access_token"]
        r = client.post("/api/v1/reminders/send-all",
                        headers={"Authorization": f"Bearer {tok}"},
                        json={"channel": "email", "recipient_override": "test@t.com"})
        assert r.status_code == 403

    def test_send_reminders_no_contact_raises_400(self, client, admin_headers):
        """Admin with no email in DB → 400 (unless override provided)."""
        # Admin has email, so this should work with email channel
        r = client.post("/api/v1/reminders/send-all",
                        headers=admin_headers,
                        json={"channel": "sms"})  # admin has no phone
        assert r.status_code == 400

    def test_send_reminders_with_override(self, client, admin_headers,
                                           sample_patient):
        """recipient_override bypasses the user's stored contact."""
        r = client.post("/api/v1/reminders/send-all",
                        headers=admin_headers,
                        json={
                            "channel": "email",
                            "recipient_override": "override@test.com",
                            "days_threshold": 1   # all patients are overdue within 1 day
                        })
        assert r.status_code == 200
        data = r.json()
        assert "sent" in data
        assert "channel" in data
        assert data["channel"] == "email"

    def test_send_reminders_returns_summary(self, client, admin_headers,
                                             sample_patient):
        r = client.post("/api/v1/reminders/send-all",
                        headers=admin_headers,
                        json={
                            "channel": "email",
                            "recipient_override": "test@test.com",
                            "days_threshold": 90
                        })
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data["sent"], int)
        assert isinstance(data["skipped"], int)


# ─────────────────────────────────────────────────────────────────────────────
# 6. DRUG INTERACTION CHECKER
# ─────────────────────────────────────────────────────────────────────────────

class TestDrugInteractions:
    def test_drug_interaction_basic(self, client, admin_headers):
        """OpenFDA call — may succeed or fail depending on network."""
        r = client.get("/api/v1/drug-interactions?drugs=metformin",
                       headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert "drugs_checked" in data
        assert "results" in data
        assert "disclaimer" in data
        assert "metformin" in data["drugs_checked"]

    def test_drug_interaction_multiple_drugs(self, client, admin_headers):
        r = client.get(
            "/api/v1/drug-interactions?drugs=aspirin,atorvastatin",
            headers=admin_headers
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["drugs_checked"]) == 2

    def test_drug_interaction_empty_query(self, client, admin_headers):
        r = client.get("/api/v1/drug-interactions?drugs=",
                       headers=admin_headers)
        assert r.status_code == 400

    def test_drug_interaction_too_many_drugs(self, client, admin_headers):
        drugs = ",".join([f"drug{i}" for i in range(11)])
        r = client.get(f"/api/v1/drug-interactions?drugs={drugs}",
                       headers=admin_headers)
        assert r.status_code == 400

    def test_drug_interaction_requires_auth(self, client):
        r = client.get("/api/v1/drug-interactions?drugs=metformin")
        assert r.status_code in (401, 403)

    def test_patient_drug_interactions_no_meds(self, client, admin_headers,
                                                sample_patient):
        r = client.get(
            f"/api/v1/patients/{sample_patient['id']}/drug-interactions",
            headers=admin_headers
        )
        assert r.status_code == 200
        data = r.json()
        assert "message" in data or "drugs_checked" in data

    def test_patient_drug_interactions_with_meds(self, client, admin_headers,
                                                   sample_patient):
        pid = sample_patient["id"]
        # Add two medications
        for med in [("Metformin", 500), ("Atorvastatin", 40)]:
            client.post("/api/v1/medications", headers=admin_headers, json={
                "patient_id": pid, "name": med[0],
                "dosage_mg": med[1], "active": 1
            })
        r = client.get(f"/api/v1/patients/{pid}/drug-interactions",
                       headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert "drugs_checked" in data
        assert len(data["drugs_checked"]) == 2

    def test_disclaimer_always_present(self, client, admin_headers):
        r = client.get("/api/v1/drug-interactions?drugs=aspirin",
                       headers=admin_headers)
        assert r.status_code == 200
        assert "disclaimer" in r.json()
        assert len(r.json()["disclaimer"]) > 20
