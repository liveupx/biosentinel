"""Tests for analytics, reports, settings, and audit log."""
import pytest


class TestAnalytics:
    def test_population_analytics_empty(self, client, admin_headers):
        r = client.get("/api/v1/analytics/population", headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["total_assessments"] == 0

    def test_population_analytics_after_predictions(self, client,
                                                     admin_headers,
                                                     sample_patient,
                                                     sample_checkup):
        client.post(f"/api/v1/patients/{sample_patient['id']}/predict",
                    headers=admin_headers)
        r = client.get("/api/v1/analytics/population", headers=admin_headers)
        data = r.json()
        assert data["total_assessments"] == 1
        assert "cancer_dist" in data
        assert "metabolic_dist" in data
        assert "cardio_dist" in data
        for d in ["cancer_dist", "metabolic_dist", "cardio_dist"]:
            dist = data[d]
            assert "low" in dist and "moderate" in dist
            assert "high" in dist and "critical" in dist
            assert "avg" in dist
            total = dist["low"] + dist["moderate"] + dist["high"] + dist["critical"]
            assert total == 1

    def test_analytics_user_scoped(self, client, admin_headers,
                                    clinician_headers):
        """Each user's analytics should only include their own patients."""
        # Admin creates + predicts
        admin_p = client.post("/api/v1/patients", headers=admin_headers,
            json={"age": 50, "sex": "Male", "family_history_cancer": 0,
                  "family_history_diabetes": 0, "family_history_cardio": 0}).json()
        client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": admin_p["id"], "checkup_date": "2024-01-01",
            "hba1c": 5.5
        })
        client.post(f"/api/v1/patients/{admin_p['id']}/predict",
                    headers=admin_headers)

        # Clinician has no patients
        clin_an = client.get("/api/v1/analytics/population",
                             headers=clinician_headers).json()
        assert clin_an["total_assessments"] == 0

        # Admin sees their prediction
        admin_an = client.get("/api/v1/analytics/population",
                              headers=admin_headers).json()
        assert admin_an["total_assessments"] == 1


class TestReport:
    def test_generate_report(self, client, admin_headers,
                              sample_patient, sample_checkup):
        pid = sample_patient["id"]
        client.post(f"/api/v1/patients/{pid}/predict",
                    headers=admin_headers)
        r = client.get(f"/api/v1/patients/{pid}/report",
                       headers=admin_headers)
        assert r.status_code == 200
        data = r.json()

        assert "patient" in data
        assert "monitoring_summary" in data
        assert "latest_risk_scores" in data
        assert "recommendation" in data
        assert "disclaimer" in data

        # Patient info correct
        assert data["patient"]["age"] == sample_patient["age"]
        assert data["patient"]["sex"] == sample_patient["sex"]

        # Monitoring summary has checkup count
        assert data["monitoring_summary"]["total_checkups"] == 1

        # Disclaimer must always be present
        assert len(data["disclaimer"]) > 50

    def test_report_without_prediction(self, client, admin_headers,
                                        sample_patient, sample_checkup):
        """Report should work even without a prediction (no risk scores)."""
        r = client.get(f"/api/v1/patients/{sample_patient['id']}/report",
                       headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["latest_risk_scores"]["cancer"] is None


class TestStats:
    def test_stats_endpoint(self, client, admin_headers):
        r = client.get("/api/v1/stats", headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        expected_keys = [
            "total_patients", "total_checkups", "total_predictions",
            "total_medications", "total_diagnoses", "unread_alerts",
            "ml_trained", "overdue_checkups"
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_stats_counts_correctly(self, client, admin_headers,
                                     sample_patient, sample_checkup):
        r = client.get("/api/v1/stats", headers=admin_headers)
        data = r.json()
        assert data["total_patients"] >= 1
        assert data["total_checkups"] >= 1

    def test_overdue_checkups_detected(self, client, admin_headers):
        """Patient with no checkups at all should count as overdue."""
        client.post("/api/v1/patients", headers=admin_headers, json={
            "age": 50, "sex": "Male",
            "family_history_cancer": 0, "family_history_diabetes": 0,
            "family_history_cardio": 0
        })
        r = client.get("/api/v1/stats", headers=admin_headers)
        assert r.json()["overdue_checkups"] >= 1


class TestEmailSettings:
    def test_get_email_config_default(self, client, admin_headers):
        r = client.get("/api/v1/settings/email", headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert "enabled" in data
        assert data["enabled"] == False   # disabled by default

    def test_save_email_config(self, client, admin_headers):
        r = client.put("/api/v1/settings/email", headers=admin_headers, json={
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_username": "test@example.com",
            "smtp_password": "apppassword123",
            "smtp_use_tls": True,
            "from_address": "alerts@example.com",
            "notify_to": "doctor@hospital.com",
            "notify_on_high": True,
            "notify_on_critical": True,
            "enabled": False   # keep disabled in tests
        })
        assert r.status_code == 200
        # Now GET and verify it was saved
        r2 = client.get("/api/v1/settings/email", headers=admin_headers)
        cfg = r2.json()
        assert cfg["smtp_host"] == "smtp.gmail.com"
        assert cfg["smtp_username"] == "test@example.com"
        assert cfg["smtp_password"] == "••••••••"  # masked


class TestAuditLog:
    def test_audit_log_records_prediction(self, client, admin_headers,
                                           sample_patient, sample_checkup):
        client.post(f"/api/v1/patients/{sample_patient['id']}/predict",
                    headers=admin_headers)
        r = client.get("/api/v1/audit-log", headers=admin_headers)
        assert r.status_code == 200
        logs = r.json()["logs"]
        assert len(logs) > 0
        actions = [l["action"] for l in logs]
        assert "run_prediction" in actions

    def test_audit_log_records_user(self, client, admin_headers,
                                     sample_patient, sample_checkup):
        client.post(f"/api/v1/patients/{sample_patient['id']}/predict",
                    headers=admin_headers)
        r = client.get("/api/v1/audit-log", headers=admin_headers)
        logs = r.json()["logs"]
        pred_logs = [l for l in logs if l["action"] == "run_prediction"]
        assert len(pred_logs) > 0
        assert pred_logs[0]["username"] == "admin"

    def test_clinician_sees_only_own_audit(self, client, admin_headers,
                                            clinician_headers):
        """Clinicians should only see their own audit entries."""
        # Admin does something
        p = client.post("/api/v1/patients", headers=admin_headers,
            json={"age": 45, "sex": "Male",
                  "family_history_cancer": 0, "family_history_diabetes": 0,
                  "family_history_cardio": 0}).json()

        # Clinician does something
        cp = client.post("/api/v1/patients", headers=clinician_headers,
            json={"age": 40, "sex": "Female",
                  "family_history_cancer": 0, "family_history_diabetes": 0,
                  "family_history_cardio": 0}).json()

        # Get clinician audit log
        r = client.get("/api/v1/audit-log", headers=clinician_headers)
        logs = r.json()["logs"]
        # All logs should belong to clinician
        for log in logs:
            assert log["username"] == "dr_test"
