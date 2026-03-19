"""
BioSentinel Python SDK
======================
A thin, typed Python client for the BioSentinel REST API.

Allows external applications to integrate with a running BioSentinel
instance — adding patients, submitting checkups, fetching predictions,
and reading alerts programmatically.

Quick start
-----------
  pip install httpx

  from biosentinel_sdk import BioSentinelClient

  client = BioSentinelClient("http://localhost:8000")
  client.login("admin", "admin123")

  # Add a patient
  patient = client.create_patient(age=52, sex="Female", ethnicity="South Asian",
                                  family_history_cancer=1)

  # Submit a checkup
  checkup = client.add_checkup(
      patient_id=patient["id"],
      checkup_date="2024-01-15",
      hba1c=6.2, glucose_fasting=114, hemoglobin=12.8,
      cea=2.3, ldl=128, bp_systolic=142, bmi=27.4
  )

  # Run AI prediction
  prediction = client.predict(patient["id"])
  print(f"Cancer risk: {prediction['cancer']['level']} ({prediction['cancer']['risk']*100:.1f}%)")

  # Get AI narrative
  narrative = client.get_narrative(patient["id"], audience="patient")
  print(narrative)
"""

from __future__ import annotations

import os
from typing import Any, Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore


class BioSentinelError(Exception):
    """Raised when the BioSentinel API returns an error."""
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class BioSentinelClient:
    """
    Authenticated client for the BioSentinel REST API.

    All methods raise BioSentinelError on API errors.
    Token is refreshed automatically on 401 responses.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        if not _HTTPX_AVAILABLE:
            raise ImportError(
                "httpx is required for the BioSentinel SDK.\n"
                "Install with: pip install httpx"
            )
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._token: Optional[str] = api_key or os.getenv("BIOSENTINEL_API_KEY")
        self._username: Optional[str] = None
        self._password: Optional[str] = None

    # ── Auth ──────────────────────────────────────────────────────────────────

    def login(self, username: str, password: str) -> dict:
        """Authenticate and store JWT token for subsequent requests."""
        self._username = username
        self._password = password
        r = httpx.post(
            f"{self.base_url}/api/v1/auth/login",
            json={"username": username, "password": password},
            timeout=self.timeout,
        )
        self._raise_for_status(r)
        data = r.json()
        self._token = data["access_token"]
        return data

    def me(self) -> dict:
        """Return current authenticated user info."""
        return self._get("/api/v1/auth/me")

    # ── Patients ──────────────────────────────────────────────────────────────

    def create_patient(
        self,
        age: int,
        sex: str,
        ethnicity: str = "",
        family_history_cancer: int = 0,
        family_history_diabetes: int = 0,
        family_history_cardio: int = 0,
        smoking_status: str = "never",
        alcohol_units_weekly: float = 0.0,
        exercise_min_weekly: float = 150.0,
        notes: str = "",
    ) -> dict:
        """Create a new patient record."""
        return self._post("/api/v1/patients", {
            "age": age, "sex": sex, "ethnicity": ethnicity,
            "family_history_cancer": family_history_cancer,
            "family_history_diabetes": family_history_diabetes,
            "family_history_cardio": family_history_cardio,
            "smoking_status": smoking_status,
            "alcohol_units_weekly": alcohol_units_weekly,
            "exercise_min_weekly": exercise_min_weekly,
            "notes": notes,
        })

    def get_patient(self, patient_id: str) -> dict:
        """Get a patient by ID."""
        return self._get(f"/api/v1/patients/{patient_id}")

    def list_patients(self, search: str = "", overdue: Optional[bool] = None) -> list[dict]:
        """List all patients. Optionally filter by search query or overdue status."""
        params: dict[str, Any] = {}
        if search:
            params["q"] = search
        if overdue is not None:
            params["overdue"] = str(overdue).lower()
        data = self._get("/api/v1/patients", params=params)
        return data.get("patients", data) if isinstance(data, dict) else data

    def delete_patient(self, patient_id: str) -> dict:
        """Delete a patient and all their data (cascades)."""
        return self._delete(f"/api/v1/patients/{patient_id}")

    # ── Checkups ──────────────────────────────────────────────────────────────

    def add_checkup(self, patient_id: str, checkup_date: str, **biomarkers) -> dict:
        """
        Add a checkup with biomarker values.

        Common biomarkers: hba1c, glucose_fasting, hemoglobin, wbc,
        lymphocytes_pct, cea, ca125, psa, alt, ast, ldl, hdl,
        triglycerides, bp_systolic, bmi, crp, creatinine, tsh, ferritin

        Example:
            client.add_checkup("pat-123", "2024-01-15",
                                hba1c=6.2, glucose_fasting=114, cea=2.3)
        """
        payload = {"patient_id": patient_id, "checkup_date": checkup_date}
        payload.update(biomarkers)
        return self._post("/api/v1/checkups", payload)

    def list_checkups(self, patient_id: str) -> list[dict]:
        """List all checkups for a patient, ordered by date."""
        data = self._get(f"/api/v1/patients/{patient_id}/checkups")
        return data.get("checkups", data) if isinstance(data, dict) else data

    def get_trends(self, patient_id: str) -> dict:
        """Get biomarker trend data (time series + slopes) for charting."""
        return self._get(f"/api/v1/patients/{patient_id}/trends")

    # ── Predictions ───────────────────────────────────────────────────────────

    def predict(self, patient_id: str) -> dict:
        """
        Run an AI risk prediction for a patient.

        Returns cancer, metabolic, cardio, hematologic risk scores with
        SHAP feature attribution and a clinical recommendation string.
        """
        return self._post(f"/api/v1/patients/{patient_id}/predict", {})

    def get_predictions(self, patient_id: str) -> list[dict]:
        """Get the full prediction history for a patient."""
        data = self._get(f"/api/v1/patients/{patient_id}/predictions")
        return data.get("predictions", data) if isinstance(data, dict) else data

    def get_shap(self, patient_id: str, domain: str = "cancer") -> dict:
        """
        Get detailed SHAP feature attribution for a domain.

        domain: "cancer" | "metabolic" | "cardio" | "hematologic"
        """
        return self._get(f"/api/v1/patients/{patient_id}/shap/{domain}")

    # ── AI Features ───────────────────────────────────────────────────────────

    def get_narrative(self, patient_id: str, audience: str = "patient") -> str:
        """
        Generate a plain-English narrative for the latest prediction.

        audience: "patient" (no jargon) | "clinician" (medical language)
        Requires ANTHROPIC_API_KEY set on the server.
        """
        data = self._post(
            f"/api/v1/ai/narrative/{patient_id}",
            {},
            params={"audience": audience},
        )
        return data.get("narrative", "")

    def get_anomalies(self, patient_id: str) -> str:
        """
        Run longitudinal anomaly detection on a patient's full timeline.
        Returns a narrative flagging subtle patterns.
        Requires ANTHROPIC_API_KEY set on the server.
        """
        data = self._get(f"/api/v1/ai/anomalies/{patient_id}")
        return data.get("analysis", "")

    def get_percentile(self, patient_id: str) -> dict:
        """
        Compare this patient's risk scores against similar-age, same-sex patients.
        Returns percentile rank per domain (0–100).
        """
        return self._get(f"/api/v1/analytics/percentile/{patient_id}")

    # ── Alerts ────────────────────────────────────────────────────────────────

    def get_alerts(self, patient_id: Optional[str] = None) -> list[dict]:
        """Get clinical alerts. If patient_id given, only that patient's alerts."""
        if patient_id:
            data = self._get(f"/api/v1/patients/{patient_id}/alerts")
        else:
            data = self._get("/api/v1/alerts")
        return data.get("alerts", data) if isinstance(data, dict) else data

    def acknowledge_alert(self, alert_id: str) -> dict:
        """Mark an alert as acknowledged."""
        return self._post(f"/api/v1/alerts/{alert_id}/acknowledge", {})

    # ── Medications ───────────────────────────────────────────────────────────

    def add_medication(
        self,
        patient_id: str,
        name: str,
        dosage_mg: float,
        frequency: str,
        start_date: str,
        prescribed_for: str = "",
        active: int = 1,
    ) -> dict:
        """Add a medication to a patient's record."""
        return self._post("/api/v1/medications", {
            "patient_id": patient_id, "name": name, "dosage_mg": dosage_mg,
            "frequency": frequency, "start_date": start_date,
            "prescribed_for": prescribed_for, "active": active,
        })

    def get_medications(self, patient_id: str) -> list[dict]:
        data = self._get(f"/api/v1/patients/{patient_id}/medications")
        return data.get("medications", data) if isinstance(data, dict) else data

    # ── Analytics ─────────────────────────────────────────────────────────────

    def population_analytics(self) -> dict:
        """Get population-level risk distribution statistics."""
        return self._get("/api/v1/analytics/population")

    def get_report(self, patient_id: str) -> dict:
        """Get the full structured patient report."""
        return self._get(f"/api/v1/patients/{patient_id}/report")

    # ── System ────────────────────────────────────────────────────────────────

    def health(self) -> dict:
        """Check API health. Does not require authentication."""
        r = httpx.get(f"{self.base_url}/health", timeout=self.timeout)
        return r.json()

    def system_info(self) -> dict:
        """Get system info: ML models, features, AI status, cache stats."""
        return self._get("/api/v1/capabilities")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    def _raise_for_status(self, response) -> None:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except Exception:
                detail = response.text
            raise BioSentinelError(response.status_code, str(detail))

    def _get(self, path: str, params: Optional[dict] = None) -> Any:
        r = httpx.get(
            f"{self.base_url}{path}",
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )
        self._raise_for_status(r)
        return r.json()

    def _post(self, path: str, body: dict, params: Optional[dict] = None) -> Any:
        r = httpx.post(
            f"{self.base_url}{path}",
            headers=self._headers(),
            json=body,
            params=params,
            timeout=self.timeout,
        )
        self._raise_for_status(r)
        return r.json()

    def _delete(self, path: str) -> Any:
        r = httpx.delete(
            f"{self.base_url}{path}",
            headers=self._headers(),
            timeout=self.timeout,
        )
        self._raise_for_status(r)
        return r.json()

    def __repr__(self) -> str:
        authed = "authenticated" if self._token else "unauthenticated"
        return f"BioSentinelClient(url={self.base_url!r}, {authed})"


# ── Convenience function ──────────────────────────────────────────────────────

def connect(
    url: str = "http://localhost:8000",
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_key: Optional[str] = None,
) -> BioSentinelClient:
    """
    Create and authenticate a BioSentinel client in one call.

    Examples::

        # With credentials
        client = connect("http://localhost:8000", "admin", "admin123")

        # With API key (set BIOSENTINEL_API_KEY env var)
        client = connect("https://biosentinel.yourhospital.com")
    """
    client = BioSentinelClient(url, api_key=api_key)
    if username and password:
        client.login(username, password)
    return client
