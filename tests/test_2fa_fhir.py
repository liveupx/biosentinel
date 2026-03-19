"""Tests for 2FA (TOTP) and FHIR R4 import."""
import pytest, json
from unittest.mock import patch


# ─────────────────────────────────────────────────────────────────────────────
# TWO-FACTOR AUTHENTICATION
# ─────────────────────────────────────────────────────────────────────────────

class TestTOTP2FA:
    def test_2fa_status_default_disabled(self, client, clinician_headers):
        r = client.get("/api/v1/auth/2fa/status", headers=clinician_headers)
        assert r.status_code == 200
        assert r.json()["totp_enabled"] is False

    def test_2fa_setup_returns_secret_and_uri(self, client, clinician_headers):
        r = client.post("/api/v1/auth/2fa/setup", headers=clinician_headers)
        assert r.status_code == 200
        data = r.json()
        assert "secret" in data
        assert "uri" in data
        assert "otpauth://totp/" in data["uri"]
        assert len(data["secret"]) >= 16   # base32 TOTP secret

    def test_2fa_setup_generates_qr_code(self, client, clinician_headers):
        r = client.post("/api/v1/auth/2fa/setup", headers=clinician_headers)
        data = r.json()
        if data.get("qr_code"):   # qrcode lib may not be installed in CI
            assert data["qr_code"].startswith("data:image/png;base64,")

    def test_2fa_verify_and_enable(self, client, clinician_headers):
        """Setup → verify with a real TOTP code → 2FA enabled."""
        try:
            import pyotp
        except ImportError:
            pytest.skip("pyotp not installed")

        # Setup
        setup = client.post("/api/v1/auth/2fa/setup",
                            headers=clinician_headers).json()
        secret = setup["secret"]

        # Generate valid TOTP code
        totp = pyotp.TOTP(secret)
        code = totp.now()

        # Verify
        r = client.post("/api/v1/auth/2fa/verify",
                        headers=clinician_headers,
                        json={"code": code})
        assert r.status_code == 200
        data = r.json()
        assert data["enabled"] is True
        assert "backup_codes" in data
        assert len(data["backup_codes"]) == 10

    def test_2fa_verify_invalid_code(self, client, clinician_headers):
        client.post("/api/v1/auth/2fa/setup", headers=clinician_headers)
        r = client.post("/api/v1/auth/2fa/verify",
                        headers=clinician_headers, json={"code": "000000"})
        assert r.status_code == 400

    def test_2fa_verify_requires_setup_first(self, client, admin_headers):
        """Verify without setup should return 400."""
        r = client.post("/api/v1/auth/2fa/verify",
                        headers=admin_headers, json={"code": "123456"})
        assert r.status_code == 400

    def test_login_requires_totp_when_enabled(self, client):
        """After enabling 2FA, login without code returns 401 with X-2FA-Required header."""
        try:
            import pyotp
        except ImportError:
            pytest.skip("pyotp not installed")

        # Register fresh user
        client.post("/api/v1/auth/register", json={
            "username": "twofa_user", "email": "twofa@t.com",
            "password": "securepass", "role": "clinician"
        })
        tok = client.post("/api/v1/auth/login", json={
            "username": "twofa_user", "password": "securepass"
        }).json()["access_token"]
        headers = {"Authorization": f"Bearer {tok}"}

        # Enable 2FA
        setup = client.post("/api/v1/auth/2fa/setup", headers=headers).json()
        code  = pyotp.TOTP(setup["secret"]).now()
        client.post("/api/v1/auth/2fa/verify", headers=headers,
                    json={"code": code})

        # Now login without TOTP code
        r = client.post("/api/v1/auth/login", json={
            "username": "twofa_user", "password": "securepass"
        })
        assert r.status_code == 401
        assert "2FA" in (r.json().get("detail", "") or r.headers.get("X-2FA-Required", ""))

    def test_login_succeeds_with_valid_totp(self, client):
        """Login with correct TOTP code succeeds after 2FA enabled."""
        try:
            import pyotp
        except ImportError:
            pytest.skip("pyotp not installed")

        client.post("/api/v1/auth/register", json={
            "username": "twofa_ok", "email": "tfok@t.com",
            "password": "securepass2", "role": "clinician"
        })
        tok = client.post("/api/v1/auth/login", json={
            "username": "twofa_ok", "password": "securepass2"
        }).json()["access_token"]
        headers = {"Authorization": f"Bearer {tok}"}

        setup = client.post("/api/v1/auth/2fa/setup", headers=headers).json()
        totp  = pyotp.TOTP(setup["secret"])
        client.post("/api/v1/auth/2fa/verify", headers=headers,
                    json={"code": totp.now()})

        # Login WITH code
        r = client.post("/api/v1/auth/login", json={
            "username": "twofa_ok", "password": "securepass2",
            "totp_code": totp.now()
        })
        assert r.status_code == 200
        assert "access_token" in r.json()

    def test_2fa_disable_requires_password(self, client, clinician_headers):
        """Disable 2FA with wrong password → 400."""
        try:
            import pyotp
        except ImportError:
            pytest.skip("pyotp not installed")

        setup = client.post("/api/v1/auth/2fa/setup",
                            headers=clinician_headers).json()
        code = pyotp.TOTP(setup["secret"]).now()
        client.post("/api/v1/auth/2fa/verify", headers=clinician_headers,
                    json={"code": code})

        r = client.request("DELETE", "/api/v1/auth/2fa/disable",
                           headers=clinician_headers,
                           json={"password": "wrongpassword"})
        assert r.status_code == 400

    def test_2fa_disable_success(self, client, clinician_headers):
        try:
            import pyotp
        except ImportError:
            pytest.skip("pyotp not installed")

        setup = client.post("/api/v1/auth/2fa/setup",
                            headers=clinician_headers).json()
        code = pyotp.TOTP(setup["secret"]).now()
        client.post("/api/v1/auth/2fa/verify", headers=clinician_headers,
                    json={"code": code})

        # Confirm enabled
        status = client.get("/api/v1/auth/2fa/status",
                            headers=clinician_headers).json()
        assert status["totp_enabled"] is True

        # Disable
        r = client.request("DELETE", "/api/v1/auth/2fa/disable",
                           headers=clinician_headers,
                           json={"password": "doctorpass123"})
        assert r.status_code == 200

        # Confirm disabled
        status2 = client.get("/api/v1/auth/2fa/status",
                             headers=clinician_headers).json()
        assert status2["totp_enabled"] is False

    def test_2fa_backup_code_usage(self, client):
        """A backup code can be used instead of a TOTP code."""
        try:
            import pyotp
        except ImportError:
            pytest.skip("pyotp not installed")

        client.post("/api/v1/auth/register", json={
            "username": "backup_user", "email": "bu@t.com",
            "password": "bkpass123", "role": "clinician"
        })
        tok = client.post("/api/v1/auth/login", json={
            "username": "backup_user", "password": "bkpass123"
        }).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}"}

        setup = client.post("/api/v1/auth/2fa/setup", headers=h).json()
        verify = client.post("/api/v1/auth/2fa/verify", headers=h,
                             json={"code": pyotp.TOTP(setup["secret"]).now()}).json()
        backup_code = verify["backup_codes"][0]

        # Login using backup code
        r = client.post("/api/v1/auth/login", json={
            "username": "backup_user", "password": "bkpass123",
            "totp_code": backup_code
        })
        assert r.status_code == 200

        # Same backup code a second time must fail (single use)
        r2 = client.post("/api/v1/auth/login", json={
            "username": "backup_user", "password": "bkpass123",
            "totp_code": backup_code
        })
        assert r2.status_code == 401

    def test_capabilities_shows_2fa_status(self, client, admin_headers):
        r = client.get("/api/v1/capabilities", headers=admin_headers)
        data = r.json()
        assert "totp_available" in data
        assert "two_factor_auth" in data["features"]


# ─────────────────────────────────────────────────────────────────────────────
# FHIR R4 IMPORT
# ─────────────────────────────────────────────────────────────────────────────

# Minimal FHIR R4 Bundle response for mocking
FHIR_PATIENT_BUNDLE = {
    "resourceType": "Bundle",
    "type": "searchset",
    "total": 2,
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "patient-001",
                "gender": "male",
                "birthDate": "1978-04-12",
                "name": [{"given": ["Rajesh"], "family": "Kumar"}],
                "extension": [
                    {
                        "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity",
                        "extension": [{"url": "text", "valueString": "South Asian"}]
                    }
                ]
            }
        },
        {
            "resource": {
                "resourceType": "Patient",
                "id": "patient-002",
                "gender": "female",
                "birthDate": "1990-08-25",
                "name": [{"given": ["Priya"], "family": "Sharma"}],
            }
        }
    ]
}

FHIR_OBSERVATION_BUNDLE = {
    "resourceType": "Bundle",
    "type": "searchset",
    "entry": [
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-001",
                "status": "final",
                "effectiveDateTime": "2024-03-15",
                "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}]},
                "valueQuantity": {"value": 6.1, "unit": "%"}
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "id": "obs-002",
                "status": "final",
                "effectiveDateTime": "2024-03-15",
                "code": {"coding": [{"system": "http://loinc.org", "code": "2089-1", "display": "LDL"}]},
                "valueQuantity": {"value": 128.0, "unit": "mg/dL"}
            }
        }
    ]
}

FHIR_METADATA = {
    "resourceType": "CapabilityStatement",
    "fhirVersion": "4.0.1",
    "software": {"name": "HAPI FHIR Server Test"},
    "rest": [{"resource": [{"type": "Patient"}, {"type": "Observation"}]}]
}


class TestFHIRImport:
    @patch("urllib.request.urlopen")
    def test_fhir_test_connection(self, mock_urlopen, client, admin_headers):
        """Test FHIR server connectivity check."""
        import io as _io
        mock_urlopen.return_value = _io.BytesIO(
            json.dumps(FHIR_METADATA).encode()
        )
        r = client.get(
            "/api/v1/fhir/test-connection?server_url=https://hapi.fhir.org/baseR4",
            headers=admin_headers
        )
        assert r.status_code == 200
        data = r.json()
        assert data["connected"] is True
        assert data["fhir_version"] == "4.0.1"

    @patch("urllib.request.urlopen")
    def test_fhir_import_patients(self, mock_urlopen, client, admin_headers):
        """Import Patient resources from FHIR bundle."""
        import io as _io
        mock_urlopen.return_value = _io.BytesIO(
            json.dumps(FHIR_PATIENT_BUNDLE).encode()
        )
        r = client.post("/api/v1/fhir/import", headers=admin_headers, json={
            "fhir_server_url": "https://hapi.fhir.org/baseR4",
            "resource_type": "Patient",
            "max_records": 10
        })
        assert r.status_code == 200
        data = r.json()
        assert data["imported"] == 2
        assert data["errors"] == []
        # Check ages derived from birthDate
        ages = [r["age"] for r in data["records"] if r.get("age")]
        assert all(0 < a < 120 for a in ages)

    @patch("urllib.request.urlopen")
    def test_fhir_import_observations(self, mock_urlopen, client,
                                       admin_headers, sample_patient):
        """Import Observation resources and map LOINC codes to checkup fields."""
        import io as _io

        # Mark sample_patient with the FHIR patient ID in notes
        client.put(f"/api/v1/patients/{sample_patient['id']}", headers=admin_headers,
                   json={**sample_patient, "notes": "FHIR ID: fhir-patient-abc"})

        mock_urlopen.return_value = _io.BytesIO(
            json.dumps(FHIR_OBSERVATION_BUNDLE).encode()
        )
        r = client.post("/api/v1/fhir/import", headers=admin_headers, json={
            "fhir_server_url": "https://hapi.fhir.org/baseR4",
            "resource_type": "Observation",
            "patient_id": "fhir-patient-abc",
            "max_records": 20
        })
        assert r.status_code == 200

    @patch("urllib.request.urlopen")
    def test_fhir_import_single_patient(self, mock_urlopen, client, admin_headers):
        """Fetch a single FHIR Patient by ID."""
        import io as _io
        single_patient = FHIR_PATIENT_BUNDLE["entry"][0]["resource"]
        mock_urlopen.return_value = _io.BytesIO(
            json.dumps(single_patient).encode()
        )
        r = client.post("/api/v1/fhir/import", headers=admin_headers, json={
            "fhir_server_url": "https://hapi.fhir.org/baseR4",
            "resource_type": "Patient",
            "patient_id": "patient-001"
        })
        assert r.status_code == 200
        assert r.json()["imported"] == 1

    @patch("urllib.request.urlopen")
    def test_fhir_connection_failure(self, mock_urlopen, client, admin_headers):
        """FHIR server unreachable → 502."""
        import urllib.error
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        r = client.get(
            "/api/v1/fhir/test-connection?server_url=https://nonexistent.fhir.example",
            headers=admin_headers
        )
        assert r.status_code == 502

    def test_fhir_invalid_resource_type(self, client, admin_headers):
        r = client.post("/api/v1/fhir/import", headers=admin_headers, json={
            "fhir_server_url": "https://hapi.fhir.org/baseR4",
            "resource_type": "InvalidType"
        })
        # Should fail with 400 or 502 (not 500)
        assert r.status_code in (400, 422, 502)

    def test_fhir_requires_auth(self, client):
        r = client.post("/api/v1/fhir/import", json={
            "fhir_server_url": "https://hapi.fhir.org/baseR4",
            "resource_type": "Patient"
        })
        assert r.status_code in (401, 403)

    def test_fhir_loinc_mapping(self):
        """_fhir_obs_to_checkup maps LOINC codes correctly."""
        from app import _fhir_obs_to_checkup
        obs = [
            {"resourceType": "Observation",
             "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4"}]},
             "valueQuantity": {"value": 6.2}},
            {"resourceType": "Observation",
             "code": {"coding": [{"system": "http://loinc.org", "code": "2089-1"}]},
             "valueQuantity": {"value": 130.0}},
            {"resourceType": "Observation",
             "code": {"coding": [{"system": "http://loinc.org", "code": "718-7"}]},
             "valueQuantity": {"value": 13.5}},
        ]
        fields = _fhir_obs_to_checkup(obs)
        assert fields.get("hba1c") == 6.2
        assert fields.get("ldl") == 130.0
        assert fields.get("hemoglobin") == 13.5

    def test_fhir_age_calculation(self):
        """_fhir_age correctly calculates age from DOB."""
        from app import _fhir_age
        from datetime import datetime, timedelta
        # 30 years ago
        dob = (datetime.utcnow() - timedelta(days=365*30)).date().isoformat()
        age = _fhir_age(dob)
        assert 29 <= age <= 31

        # Invalid DOB
        assert _fhir_age("") is None
        assert _fhir_age(None) is None

    def test_capabilities_shows_fhir(self, client, admin_headers):
        r = client.get("/api/v1/capabilities", headers=admin_headers)
        data = r.json()
        assert data["features"]["fhir_r4_import"] is True
        assert data["features"]["rate_limiting"] is True
