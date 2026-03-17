"""Tests for patient endpoints including multi-user isolation."""
import pytest


PATIENT_PAYLOAD = {
    "age": 45, "sex": "Female", "ethnicity": "Caucasian",
    "family_history_cancer": 1, "family_history_diabetes": 0,
    "family_history_cardio": 1, "smoking_status": "former",
    "alcohol_units_weekly": 3.0, "exercise_min_weekly": 120,
    "notes": "Test patient"
}


class TestPatientCRUD:
    def test_create_patient(self, client, admin_headers):
        r = client.post("/api/v1/patients", headers=admin_headers,
                        json=PATIENT_PAYLOAD)
        assert r.status_code == 201
        data = r.json()
        assert data["age"] == 45
        assert data["sex"] == "Female"
        assert "id" in data
        assert "created_at" in data

    def test_patient_id_is_uuid(self, client, admin_headers):
        r = client.post("/api/v1/patients", headers=admin_headers,
                        json=PATIENT_PAYLOAD)
        pid = r.json()["id"]
        # UUID format: 8-4-4-4-12 chars
        parts = pid.split("-")
        assert len(parts) == 5
        assert len(pid) == 36

    def test_get_patient(self, client, admin_headers, sample_patient):
        r = client.get(f"/api/v1/patients/{sample_patient['id']}",
                       headers=admin_headers)
        assert r.status_code == 200
        assert r.json()["id"] == sample_patient["id"]

    def test_get_nonexistent_patient(self, client, admin_headers):
        r = client.get("/api/v1/patients/nonexistent-id",
                       headers=admin_headers)
        assert r.status_code == 404

    def test_list_patients(self, client, admin_headers):
        # Create 3 patients
        for i in range(3):
            client.post("/api/v1/patients", headers=admin_headers,
                        json={**PATIENT_PAYLOAD, "age": 40 + i})
        r = client.get("/api/v1/patients", headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert data["total"] == 3
        assert len(data["patients"]) == 3

    def test_update_patient(self, client, admin_headers, sample_patient):
        updated = {**PATIENT_PAYLOAD, "age": 50, "notes": "Updated notes"}
        r = client.put(f"/api/v1/patients/{sample_patient['id']}",
                       headers=admin_headers, json=updated)
        assert r.status_code == 200
        assert r.json()["age"] == 50
        assert r.json()["notes"] == "Updated notes"

    def test_delete_patient(self, client, admin_headers, sample_patient):
        pid = sample_patient["id"]
        r = client.delete(f"/api/v1/patients/{pid}", headers=admin_headers)
        assert r.status_code == 200
        # Verify deleted
        r2 = client.get(f"/api/v1/patients/{pid}", headers=admin_headers)
        assert r2.status_code == 404

    def test_delete_cascades_checkups(self, client, admin_headers,
                                      sample_patient, sample_checkup):
        """Deleting a patient should delete all their checkups too."""
        pid = sample_patient["id"]
        # Verify checkup exists
        r = client.get(f"/api/v1/patients/{pid}/checkups",
                       headers=admin_headers)
        assert r.json()["count"] == 1
        # Delete patient
        client.delete(f"/api/v1/patients/{pid}", headers=admin_headers)
        # Patient gone → checkup gone (cascade)
        r2 = client.get(f"/api/v1/patients/{pid}/checkups",
                        headers=admin_headers)
        assert r2.status_code == 404


class TestMultiUserIsolation:
    """
    CRITICAL SECURITY TESTS: verify that one user cannot access
    another user's patients.
    """

    def test_clinician_only_sees_own_patients(self, client,
                                               admin_headers,
                                               clinician_headers):
        # Admin creates a patient
        admin_p = client.post("/api/v1/patients", headers=admin_headers,
                              json=PATIENT_PAYLOAD).json()
        # Clinician creates their own patient
        clin_p = client.post("/api/v1/patients", headers=clinician_headers,
                             json={**PATIENT_PAYLOAD, "age": 35}).json()

        # Admin sees both
        admin_list = client.get("/api/v1/patients", headers=admin_headers)
        assert admin_list.json()["total"] == 2

        # Clinician sees only their own
        clin_list = client.get("/api/v1/patients", headers=clinician_headers)
        assert clin_list.json()["total"] == 1
        assert clin_list.json()["patients"][0]["id"] == clin_p["id"]

    def test_cannot_access_other_users_patient_by_id(self, client,
                                                      admin_headers,
                                                      clinician_headers):
        """Clinician trying to GET admin's patient must get 403."""
        admin_p = client.post("/api/v1/patients", headers=admin_headers,
                              json=PATIENT_PAYLOAD).json()
        r = client.get(f"/api/v1/patients/{admin_p['id']}",
                       headers=clinician_headers)
        assert r.status_code == 403

    def test_cannot_add_checkup_to_other_users_patient(self, client,
                                                         admin_headers,
                                                         clinician_headers,
                                                         sample_patient):
        """Clinician trying to add a checkup to admin's patient must get 403."""
        r = client.post("/api/v1/checkups", headers=clinician_headers, json={
            "patient_id": sample_patient["id"],
            "checkup_date": "2024-01-01",
            "hba1c": 5.5
        })
        assert r.status_code == 403

    def test_cannot_delete_other_users_patient(self, client,
                                                admin_headers,
                                                clinician_headers,
                                                sample_patient):
        r = client.delete(f"/api/v1/patients/{sample_patient['id']}",
                          headers=clinician_headers)
        assert r.status_code == 403
        # Patient still exists for admin
        r2 = client.get(f"/api/v1/patients/{sample_patient['id']}",
                        headers=admin_headers)
        assert r2.status_code == 200

    def test_cannot_predict_on_other_users_patient(self, client,
                                                    admin_headers,
                                                    clinician_headers,
                                                    sample_patient,
                                                    sample_checkup):
        r = client.post(f"/api/v1/patients/{sample_patient['id']}/predict",
                        headers=clinician_headers)
        assert r.status_code == 403

    def test_admin_can_access_all_patients(self, client,
                                           admin_headers,
                                           clinician_headers):
        """Admin role bypasses isolation and can access any patient."""
        clin_p = client.post("/api/v1/patients", headers=clinician_headers,
                             json=PATIENT_PAYLOAD).json()
        # Admin CAN access clinician's patient
        r = client.get(f"/api/v1/patients/{clin_p['id']}",
                       headers=admin_headers)
        assert r.status_code == 200

    def test_owner_id_set_correctly(self, client, clinician_headers,
                                    clinician_token):
        """Patient's owner_id must equal the creating user's id."""
        # Get clinician's user id
        me = client.get("/api/v1/auth/me", headers=clinician_headers).json()
        p = client.post("/api/v1/patients", headers=clinician_headers,
                        json=PATIENT_PAYLOAD).json()
        assert p["owner_id"] == me["id"]
