"""Tests for medications, diagnoses, and diet plans."""
import pytest


class TestMedications:
    def test_create_medication(self, client, admin_headers, sample_patient):
        r = client.post("/api/v1/medications", headers=admin_headers, json={
            "patient_id": sample_patient["id"],
            "name": "Metformin",
            "dosage_mg": 500,
            "frequency": "Twice daily",
            "start_date": "2024-01-01",
            "prescribed_for": "Pre-diabetes",
            "active": 1
        })
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "Metformin"
        assert data["dosage_mg"] == 500
        assert data["active"] == 1

    def test_list_medications(self, client, admin_headers, sample_patient):
        pid = sample_patient["id"]
        meds = ["Aspirin", "Atorvastatin", "Levothyroxine"]
        for m in meds:
            client.post("/api/v1/medications", headers=admin_headers, json={
                "patient_id": pid, "name": m, "active": 1
            })
        r = client.get(f"/api/v1/patients/{pid}/medications",
                       headers=admin_headers)
        assert r.status_code == 200
        assert len(r.json()["medications"]) == 3

    def test_delete_medication(self, client, admin_headers, sample_patient):
        pid = sample_patient["id"]
        med = client.post("/api/v1/medications", headers=admin_headers, json={
            "patient_id": pid, "name": "Aspirin", "active": 1
        }).json()
        r = client.delete(f"/api/v1/medications/{med['id']}",
                          headers=admin_headers)
        assert r.status_code == 200
        meds = client.get(f"/api/v1/patients/{pid}/medications",
                          headers=admin_headers).json()["medications"]
        assert len(meds) == 0

    def test_cannot_add_medication_to_other_users_patient(
            self, client, admin_headers, clinician_headers, sample_patient):
        r = client.post("/api/v1/medications", headers=clinician_headers, json={
            "patient_id": sample_patient["id"],
            "name": "Aspirin", "active": 1
        })
        assert r.status_code == 403


class TestDiagnoses:
    def test_create_diagnosis(self, client, admin_headers, sample_patient):
        r = client.post("/api/v1/diagnoses", headers=admin_headers, json={
            "patient_id": sample_patient["id"],
            "icd10_code": "E11.9",
            "description": "Type 2 diabetes mellitus without complications",
            "diagnosed_date": "2024-01-15",
            "status": "active",
            "severity": "moderate"
        })
        assert r.status_code == 201
        data = r.json()
        assert data["icd10_code"] == "E11.9"
        assert data["status"] == "active"

    def test_list_diagnoses(self, client, admin_headers, sample_patient):
        pid = sample_patient["id"]
        for code, desc in [("E11.9", "Diabetes"), ("I10", "Hypertension")]:
            client.post("/api/v1/diagnoses", headers=admin_headers, json={
                "patient_id": pid, "icd10_code": code,
                "description": desc, "status": "active"
            })
        r = client.get(f"/api/v1/patients/{pid}/diagnoses",
                       headers=admin_headers)
        assert len(r.json()["diagnoses"]) == 2

    def test_delete_diagnosis(self, client, admin_headers, sample_patient):
        pid = sample_patient["id"]
        diag = client.post("/api/v1/diagnoses", headers=admin_headers, json={
            "patient_id": pid, "description": "Hypertension", "status": "active"
        }).json()
        r = client.delete(f"/api/v1/diagnoses/{diag['id']}",
                          headers=admin_headers)
        assert r.status_code == 200


class TestDietPlans:
    def test_create_diet_plan(self, client, admin_headers, sample_patient):
        r = client.post("/api/v1/diet-plans", headers=admin_headers, json={
            "patient_id": sample_patient["id"],
            "start_date": "2024-01-01",
            "calories_daily": 1800,
            "protein_g": 75,
            "carbs_g": 220,
            "fat_g": 65,
            "fiber_g": 28,
            "diet_type": "mediterranean",
            "notes": "Low-GI Mediterranean diet"
        })
        assert r.status_code == 201
        data = r.json()
        assert data["diet_type"] == "mediterranean"
        assert data["calories_daily"] == 1800

    def test_list_diet_plans(self, client, admin_headers, sample_patient):
        pid = sample_patient["id"]
        for i, diet_type in enumerate(["balanced", "low_carb"]):
            client.post("/api/v1/diet-plans", headers=admin_headers, json={
                "patient_id": pid,
                "start_date": f"202{i+3}-01-01",
                "diet_type": diet_type,
                "calories_daily": 1800
            })
        r = client.get(f"/api/v1/patients/{pid}/diet-plans",
                       headers=admin_headers)
        assert r.status_code == 200
        assert len(r.json()["diet_plans"]) == 2
