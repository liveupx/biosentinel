"""Tests for checkup ingestion and biomarker tracking."""
import pytest


FULL_CHECKUP = {
    "checkup_date": "2024-01-15",
    "weight_kg": 71.2, "bmi": 26.8,
    "bp_systolic": 126, "bp_diastolic": 80, "heart_rate": 72,
    "wbc": 7.4, "hemoglobin": 13.5, "platelets": 252,
    "lymphocytes_pct": 32, "neutrophils_pct": 60,
    "glucose_fasting": 96, "hba1c": 5.5, "creatinine": 0.82,
    "alt": 24, "ast": 21, "albumin": 4.3, "crp": 1.2,
    "total_cholesterol": 192, "ldl": 114, "hdl": 54, "triglycerides": 122,
    "tsh": 2.3, "vitamin_d": 26.0, "ferritin": 48.0,
    "cea": 1.8, "ca125": 14.2, "psa": None,
    "notes": "Routine quarterly checkup"
}


class TestCheckupCRUD:
    def test_create_checkup_full(self, client, admin_headers, sample_patient):
        r = client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": sample_patient["id"], **FULL_CHECKUP
        })
        assert r.status_code == 201
        data = r.json()
        assert data["hba1c"] == 5.5
        assert data["cea"] == 1.8
        assert data["checkup_date"] == "2024-01-15"
        assert data["notes"] == "Routine quarterly checkup"

    def test_create_checkup_partial_fields(self, client, admin_headers,
                                            sample_patient):
        """Only required fields — all labs are optional."""
        r = client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": sample_patient["id"],
            "checkup_date": "2024-04-01",
            "hba1c": 5.8,   # just one lab value
        })
        assert r.status_code == 201
        data = r.json()
        assert data["hba1c"] == 5.8
        assert data["cea"] is None   # unset fields are null

    def test_checkup_requires_existing_patient(self, client, admin_headers):
        r = client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": "nonexistent-patient-id",
            "checkup_date": "2024-01-01"
        })
        assert r.status_code in (403, 404)

    def test_list_checkups_sorted_by_date(self, client, admin_headers,
                                           sample_patient):
        pid = sample_patient["id"]
        dates = ["2024-07-01", "2023-01-01", "2024-01-01", "2023-07-01"]
        for d in dates:
            client.post("/api/v1/checkups", headers=admin_headers, json={
                "patient_id": pid, "checkup_date": d, "hba1c": 5.5
            })
        r = client.get(f"/api/v1/patients/{pid}/checkups",
                       headers=admin_headers)
        chks = r.json()["checkups"]
        assert len(chks) == 4
        # Must be ascending by date
        returned_dates = [c["checkup_date"] for c in chks]
        assert returned_dates == sorted(returned_dates)

    def test_delete_checkup(self, client, admin_headers, sample_checkup):
        cid = sample_checkup["id"]
        r = client.delete(f"/api/v1/checkups/{cid}", headers=admin_headers)
        assert r.status_code == 200
        r2 = client.get(f"/api/v1/checkups/{cid}", headers=admin_headers)
        assert r2.status_code == 404

    def test_checkup_count_in_response(self, client, admin_headers,
                                        sample_patient):
        pid = sample_patient["id"]
        for i in range(5):
            client.post("/api/v1/checkups", headers=admin_headers, json={
                "patient_id": pid,
                "checkup_date": f"202{i+1}-01-01",
                "hba1c": 5.0 + i * 0.1
            })
        r = client.get(f"/api/v1/patients/{pid}/checkups",
                       headers=admin_headers)
        assert r.json()["count"] == 5


class TestBiomarkerTrends:
    def test_trends_empty_no_checkups(self, client, admin_headers,
                                       sample_patient):
        r = client.get(f"/api/v1/patients/{sample_patient['id']}/trends",
                       headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["labels"] == []
        assert r.json()["series"] == {}

    def test_trends_populated_after_checkups(self, client, admin_headers,
                                              sample_patient):
        pid = sample_patient["id"]
        for i in range(3):
            client.post("/api/v1/checkups", headers=admin_headers, json={
                "patient_id": pid,
                "checkup_date": f"2024-0{i+1}-01",
                "hba1c": 5.5 + i * 0.1,
                "cea": 1.5 + i * 0.3
            })
        r = client.get(f"/api/v1/patients/{pid}/trends",
                       headers=admin_headers)
        data = r.json()
        assert len(data["labels"]) == 3
        assert "hba1c" in data["series"]
        assert "cea" in data["series"]

    def test_trend_direction_up(self, client, admin_headers, sample_patient):
        """Rising values should be flagged as 'up' trend."""
        pid = sample_patient["id"]
        for i, v in enumerate([5.5, 5.7, 5.9, 6.1]):
            client.post("/api/v1/checkups", headers=admin_headers, json={
                "patient_id": pid,
                "checkup_date": f"2024-0{i+1}-01",
                "hba1c": v
            })
        r = client.get(f"/api/v1/patients/{pid}/trends",
                       headers=admin_headers)
        assert r.json()["series"]["hba1c"]["trend"] == "up"

    def test_trend_status_high(self, client, admin_headers, sample_patient):
        """CEA > 5.0 should be flagged as 'high' status."""
        pid = sample_patient["id"]
        client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": pid, "checkup_date": "2024-01-01",
            "cea": 7.5   # above 5.0 threshold
        })
        r = client.get(f"/api/v1/patients/{pid}/trends",
                       headers=admin_headers)
        assert r.json()["series"]["cea"]["status"] == "high"
        assert r.json()["series"]["cea"]["latest"] == 7.5

    def test_trend_reference_ranges_present(self, client, admin_headers,
                                             sample_patient):
        """Every tracked biomarker should have a reference range."""
        pid = sample_patient["id"]
        client.post("/api/v1/checkups", headers=admin_headers, json={
            "patient_id": pid, "checkup_date": "2024-01-01",
            "hba1c": 5.5, "ldl": 110, "crp": 1.5
        })
        r = client.get(f"/api/v1/patients/{pid}/trends",
                       headers=admin_headers)
        for field in ["hba1c", "ldl", "crp"]:
            if field in r.json()["series"]:
                ref = r.json()["series"][field]["ref"]
                assert "lo" in ref
                assert "hi" in ref
                assert "unit" in ref
