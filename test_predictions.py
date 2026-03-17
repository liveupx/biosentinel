"""Tests for AI prediction engine, alerts, and trajectory."""
import pytest


class TestPrediction:
    def test_predict_requires_checkup(self, client, admin_headers,
                                       sample_patient):
        """Prediction with zero checkups should return 400."""
        r = client.post(
            f"/api/v1/patients/{sample_patient['id']}/predict",
            headers=admin_headers
        )
        assert r.status_code == 400
        assert "checkup" in r.json()["detail"].lower()

    def test_predict_success(self, client, admin_headers,
                              sample_patient, sample_checkup):
        r = client.post(
            f"/api/v1/patients/{sample_patient['id']}/predict",
            headers=admin_headers
        )
        assert r.status_code == 200
        data = r.json()

        # All 4 domains present
        for domain in ["cancer", "metabolic", "cardio", "hematologic"]:
            assert domain in data
            assert "risk" in data[domain]
            assert "level" in data[domain]

        # Risk scores are calibrated floats 0–1
        for domain in ["cancer", "metabolic", "cardio", "hematologic"]:
            risk = data[domain]["risk"]
            assert 0.0 <= risk <= 1.0, f"{domain} risk {risk} out of range"

        # Level must be one of the four categories
        valid_levels = {"LOW", "MODERATE", "HIGH", "CRITICAL"}
        for domain in ["cancer", "metabolic", "cardio", "hematologic"]:
            assert data[domain]["level"] in valid_levels

        # Composite score exists and is in range
        assert 0.0 <= data["composite"] <= 1.0

        # Metadata present
        assert data["checkups_used"] == 1
        assert "recommendation" in data
        assert len(data["recommendation"]) > 10

    def test_predict_returns_top_features(self, client, admin_headers,
                                           sample_patient, sample_checkup):
        r = client.post(
            f"/api/v1/patients/{sample_patient['id']}/predict",
            headers=admin_headers
        )
        data = r.json()
        assert "top_features" in data
        assert isinstance(data["top_features"], list)
        # Each feature has required fields
        for feat in data["top_features"]:
            assert "label" in feat
            assert "impact" in feat
            assert "direction" in feat
            assert feat["direction"] in ("risk_increasing", "protective")

    def test_predict_more_checkups_same_patient(self, client, admin_headers,
                                                  sample_patient):
        """Adding more checkups should use all of them."""
        pid = sample_patient["id"]
        dates_and_vals = [
            ("2022-01-01", 5.5, 1.8), ("2022-07-01", 5.6, 2.1),
            ("2023-01-01", 5.8, 2.6), ("2023-07-01", 5.9, 3.1),
            ("2024-01-01", 6.1, 3.8),
        ]
        for date, hba1c, cea in dates_and_vals:
            client.post("/api/v1/checkups", headers=admin_headers, json={
                "patient_id": pid, "checkup_date": date,
                "hba1c": hba1c, "cea": cea,
                "hemoglobin": 13.5, "lymphocytes_pct": 30
            })
        r = client.post(f"/api/v1/patients/{pid}/predict",
                        headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["checkups_used"] == 5

    def test_healthy_patient_low_risk(self, client, admin_headers):
        """A young, healthy patient with no risk factors should be LOW risk."""
        # Create healthy young patient
        p = client.post("/api/v1/patients", headers=admin_headers, json={
            "age": 28, "sex": "Male",
            "family_history_cancer": 0, "family_history_diabetes": 0,
            "family_history_cardio": 0, "smoking_status": "never",
            "alcohol_units_weekly": 0, "exercise_min_weekly": 300
        }).json()
        # Add clean checkup
        client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": p["id"], "checkup_date": "2024-01-01",
            "hba1c": 5.0, "glucose_fasting": 82, "hemoglobin": 15.5,
            "lymphocytes_pct": 35, "wbc": 6.5, "cea": 1.0,
            "alt": 15, "ldl": 85, "hdl": 65, "bp_systolic": 110, "bmi": 22.0,
            "crp": 0.4
        })
        r = client.post(f"/api/v1/patients/{p['id']}/predict",
                        headers=admin_headers)
        data = r.json()
        # Not guaranteed LOW but should be lower than high-risk patient
        assert data["composite"] < 0.5, \
            f"Healthy patient composite risk {data['composite']:.2f} seems too high"

    def test_prediction_saved_to_db(self, client, admin_headers,
                                     sample_patient, sample_checkup):
        """Predictions should be persisted and retrievable."""
        pid = sample_patient["id"]
        client.post(f"/api/v1/patients/{pid}/predict",
                    headers=admin_headers)
        r = client.get(f"/api/v1/patients/{pid}/predictions",
                       headers=admin_headers)
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) == 1
        assert preds[0]["checkups_used"] == 1


class TestAlerts:
    def test_alerts_generated_for_high_risk(self, client, admin_headers,
                                             sample_patient):
        """High-risk checkup data should generate alerts."""
        pid = sample_patient["id"]
        # Dangerous values
        client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": pid, "checkup_date": "2024-01-01",
            "hba1c": 7.5,           # diabetic
            "cea": 12.0,            # very elevated tumor marker
            "lymphocytes_pct": 12,  # critically low
            "bp_systolic": 160      # stage 2 hypertension
        })
        r = client.post(f"/api/v1/patients/{pid}/predict",
                        headers=admin_headers)
        assert r.status_code == 200
        alerts = r.json().get("alerts", [])
        assert len(alerts) > 0
        # Verify alert structure
        for a in alerts:
            assert "level" in a
            assert a["level"] in ("CRITICAL", "WARNING", "INFO")
            assert "cat" in a
            assert "msg" in a
            assert len(a["msg"]) > 10

    def test_alerts_saved_to_db(self, client, admin_headers, sample_patient):
        pid = sample_patient["id"]
        client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": pid, "checkup_date": "2024-01-01",
            "hba1c": 7.5, "cea": 12.0
        })
        client.post(f"/api/v1/patients/{pid}/predict",
                    headers=admin_headers)
        r = client.get(f"/api/v1/patients/{pid}/alerts",
                       headers=admin_headers)
        assert r.status_code == 200
        assert len(r.json()["alerts"]) > 0

    def test_acknowledge_alert(self, client, admin_headers, sample_patient):
        pid = sample_patient["id"]
        client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": pid, "checkup_date": "2024-01-01",
            "hba1c": 7.5, "cea": 12.0
        })
        client.post(f"/api/v1/patients/{pid}/predict",
                    headers=admin_headers)
        # Get alerts
        alerts = client.get(f"/api/v1/patients/{pid}/alerts",
                            headers=admin_headers).json()["alerts"]
        assert len(alerts) > 0
        # Acknowledge first alert
        aid = alerts[0]["id"]
        r = client.post(f"/api/v1/alerts/{aid}/acknowledge",
                        headers=admin_headers)
        assert r.status_code == 200
        # Verify acknowledged
        updated = client.get(f"/api/v1/patients/{pid}/alerts",
                             headers=admin_headers).json()["alerts"]
        acked = next(a for a in updated if a["id"] == aid)
        assert acked["acknowledged"] == 1

    def test_unread_alert_count_in_stats(self, client, admin_headers,
                                          sample_patient):
        pid = sample_patient["id"]
        client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": pid, "checkup_date": "2024-01-01",
            "cea": 12.0, "hba1c": 7.5
        })
        client.post(f"/api/v1/patients/{pid}/predict",
                    headers=admin_headers)
        stats = client.get("/api/v1/stats", headers=admin_headers).json()
        assert stats["unread_alerts"] > 0


class TestRiskTrajectory:
    def test_trajectory_empty_no_predictions(self, client, admin_headers,
                                              sample_patient):
        r = client.get(
            f"/api/v1/analytics/risk-trajectory/{sample_patient['id']}",
            headers=admin_headers
        )
        assert r.status_code == 200
        data = r.json()
        assert data["labels"] == []

    def test_trajectory_accumulates_over_time(self, client, admin_headers,
                                               sample_patient):
        pid = sample_patient["id"]
        # Add checkups over time and run multiple predictions
        for i, (date, hba1c, cea) in enumerate([
            ("2023-01-01", 5.6, 2.0),
            ("2023-07-01", 5.8, 2.5),
            ("2024-01-01", 6.0, 3.0),
        ]):
            client.post("/api/v1/checkups", headers=admin_headers, json={
                "patient_id": pid, "checkup_date": date,
                "hba1c": hba1c, "cea": cea
            })
            client.post(f"/api/v1/patients/{pid}/predict",
                        headers=admin_headers)

        r = client.get(
            f"/api/v1/analytics/risk-trajectory/{pid}",
            headers=admin_headers
        )
        data = r.json()
        assert len(data["labels"]) == 3
        assert len(data["cancer"]) == 3
        assert len(data["metabolic"]) == 3
        assert len(data["composite"]) == 3
        # All values in valid range
        for v in data["cancer"] + data["metabolic"] + data["composite"]:
            assert 0.0 <= v <= 1.0
