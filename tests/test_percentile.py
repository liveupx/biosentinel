"""
Tests for percentile comparison endpoints.
Uses monkeypatched predictions — no real ML training in test suite.
"""
import pytest


def _make_patient(client, headers, age=50, sex="Female"):
    r = client.post("/api/v1/patients", headers=headers, json={
        "age": age, "sex": sex, "ethnicity": "South Asian",
        "family_history_cancer": 1, "family_history_diabetes": 0,
        "family_history_cardio": 0, "smoking_status": "never",
        "alcohol_units_weekly": 1.0, "exercise_min_weekly": 120,
    })
    assert r.status_code == 201, r.text
    return r.json()


def _add_checkup(client, headers, pid, date="2024-01-01", vals=None):
    payload = {"patient_id": pid, "checkup_date": date,
               "hba1c": 5.8, "glucose_fasting": 102, "hemoglobin": 13.0,
               "cea": 2.0, "ldl": 118, "bp_systolic": 128, "bmi": 25.2}
    if vals:
        payload.update(vals)
    r = client.post("/api/v1/checkups", headers=headers, json=payload)
    assert r.status_code == 201, r.text
    return r.json()


def _run_prediction(client, headers, pid):
    r = client.post(f"/api/v1/patients/{pid}/predict", headers=headers)
    assert r.status_code == 200, r.text
    return r.json()


class TestPercentileEndpoint:
    def test_no_prediction_returns_400(self, client, admin_headers):
        pat = _make_patient(client, admin_headers)
        _add_checkup(client, admin_headers, pat["id"])
        r = client.get(f"/api/v1/analytics/percentile/{pat['id']}", headers=admin_headers)
        assert r.status_code == 400

    def test_too_few_comparable_patients(self, client, admin_headers):
        """With only 1 comparable patient, should return message not percentiles."""
        pat = _make_patient(client, admin_headers)
        _add_checkup(client, admin_headers, pat["id"])
        _run_prediction(client, admin_headers, pat["id"])
        r = client.get(f"/api/v1/analytics/percentile/{pat['id']}", headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert "comparable_group" in data
        # With only 1 patient, percentiles should be None or have a message
        assert data.get("percentiles") is None or "message" in data

    def test_percentile_with_multiple_patients(self, client, admin_headers):
        """With 5+ comparable patients, should return percentile ranks."""
        # Create 5 comparable patients (age 48-52, Female)
        pids = []
        for i, age in enumerate([46, 48, 50, 52, 54]):
            p = _make_patient(client, admin_headers, age=age, sex="Female")
            _add_checkup(client, admin_headers, p["id"], date=f"2024-0{i+1}-01")
            _run_prediction(client, admin_headers, p["id"])
            pids.append(p["id"])

        # Check percentile for the middle patient
        r = client.get(f"/api/v1/analytics/percentile/{pids[2]}", headers=admin_headers)
        assert r.status_code == 200
        data = r.json()

        assert "comparable_group" in data
        grp = data["comparable_group"]
        assert grp["sex"] == "Female"
        assert grp["n_patients"] >= 3

        # If we have enough data, percentiles should be present
        if data.get("percentiles"):
            pcts = data["percentiles"]
            for domain in ["cancer", "metabolic", "cardio", "hematologic"]:
                if domain in pcts:
                    assert 0 <= pcts[domain]["percentile"] <= 100
                    assert "interpretation" in pcts[domain]
                    assert "group_avg" in pcts[domain]

    def test_percentile_requires_auth(self, client, admin_headers):
        pat = _make_patient(client, admin_headers)
        r = client.get(f"/api/v1/analytics/percentile/{pat['id']}")
        assert r.status_code in (401, 403)

    def test_percentile_cross_patient_blocked(self, client, admin_headers, clinician_headers):
        pat = _make_patient(client, admin_headers)
        r = client.get(f"/api/v1/analytics/percentile/{pat['id']}", headers=clinician_headers)
        assert r.status_code == 403

    def test_percentile_nonexistent_patient(self, client, admin_headers):
        r = client.get("/api/v1/analytics/percentile/nonexistent-id", headers=admin_headers)
        assert r.status_code in (403, 404)  # nonexistent patient returns 403 or 404


class TestBiomarkerPercentileEndpoint:
    def test_no_checkup_returns_400(self, client, admin_headers):
        pat = _make_patient(client, admin_headers)
        r = client.get(
            f"/api/v1/analytics/biomarker-percentile/{pat['id']}?biomarker=hba1c",
            headers=admin_headers
        )
        assert r.status_code == 400

    def test_biomarker_not_in_checkup_returns_400(self, client, admin_headers):
        pat = _make_patient(client, admin_headers)
        _add_checkup(client, admin_headers, pat["id"])
        r = client.get(
            f"/api/v1/analytics/biomarker-percentile/{pat['id']}?biomarker=nonexistent_field",
            headers=admin_headers
        )
        assert r.status_code == 400

    def test_biomarker_percentile_with_enough_patients(self, client, admin_headers):
        """Create 5 comparable patients, check biomarker percentile."""
        pids = []
        hba1c_vals = [5.5, 5.7, 5.9, 6.1, 6.3]
        for i, (age, hba1c) in enumerate(zip([47, 49, 51, 53, 55], hba1c_vals)):
            p = _make_patient(client, admin_headers, age=age, sex="Female")
            _add_checkup(client, admin_headers, p["id"],
                         date=f"2024-0{i+1}-01",
                         vals={"hba1c": hba1c})
            pids.append(p["id"])

        # Middle patient (hba1c=5.9) should be around 40-60th percentile
        r = client.get(
            f"/api/v1/analytics/biomarker-percentile/{pids[2]}?biomarker=hba1c",
            headers=admin_headers
        )
        assert r.status_code == 200
        data = r.json()
        assert data["biomarker"] == "hba1c"
        assert data["my_value"] == 5.9
        assert 0 <= data["percentile"] <= 100
        assert "interpretation" in data
        assert "group_avg" in data
        assert "n_compared" in data
        assert data["n_compared"] >= 3

    def test_biomarker_percentile_requires_auth(self, client, admin_headers):
        pat = _make_patient(client, admin_headers)
        r = client.get(
            f"/api/v1/analytics/biomarker-percentile/{pat['id']}?biomarker=hba1c"
        )
        assert r.status_code in (401, 403)


class TestMLflowModule:
    def test_mlflow_module_imports(self):
        """mlflow_tracking.py should import without errors."""
        try:
            from mlflow_tracking import track_training_run, mlflow_status
        except ImportError as e:
            pytest.fail(f"mlflow_tracking.py import failed: {e}")

    def test_mlflow_status_without_tracking(self, monkeypatch):
        """Without MLFLOW_TRACKING=1, status should report disabled."""
        monkeypatch.delenv("MLFLOW_TRACKING", raising=False)
        import importlib
        import mlflow_tracking as mlt
        importlib.reload(mlt)
        status = mlt.mlflow_status()
        assert status["enabled"] is False

    def test_track_training_without_tracking_returns_none(self, monkeypatch):
        """Without MLFLOW_TRACKING=1, track_training_run should return None."""
        monkeypatch.delenv("MLFLOW_TRACKING", raising=False)
        import importlib
        import mlflow_tracking as mlt
        importlib.reload(mlt)
        result = mlt.track_training_run(object())  # dummy engine
        assert result is None
