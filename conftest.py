"""
BioSentinel Test Fixtures — temp-file SQLite so tables persist across connections
"""
import os, tempfile, pytest

_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
TEST_DB_PATH = _tmp.name
os.environ["DATABASE_URL"] = f"sqlite:///{TEST_DB_PATH}"
os.environ["SECRET_KEY"]    = "test-secret-not-for-prod"
os.environ["EMAIL_ENABLED"] = "false"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

import app as app_module
from app import app, Base, get_db

TEST_ENGINE    = create_engine(f"sqlite:///{TEST_DB_PATH}",
                               connect_args={"check_same_thread": False})
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)
app_module.engine       = TEST_ENGINE
app_module.SessionLocal = TestingSession

def override_get_db():
    db = TestingSession()
    try:    yield db
    finally: db.close()

@pytest.fixture(scope="session", autouse=True)
def setup_db():
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)
    try: os.unlink(TEST_DB_PATH)
    except OSError: pass

@pytest.fixture(autouse=True)
def clean_db(setup_db):
    yield
    db = TestingSession()
    try:
        for t in reversed(Base.metadata.sorted_tables):
            db.execute(t.delete())
        db.commit()
    finally:
        db.close()

@pytest.fixture
def client(clean_db):
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()

@pytest.fixture
def admin_token(client):
    r = client.post("/api/v1/auth/register", json={
        "username": "admin", "email": "a@t.com",
        "password": "adminpass123", "role": "admin"})
    assert r.status_code == 201, r.text
    return r.json()["access_token"]

@pytest.fixture
def clinician_token(client):
    r = client.post("/api/v1/auth/register", json={
        "username": "dr_test", "email": "dr@t.com",
        "password": "doctorpass123", "role": "clinician"})
    assert r.status_code == 201, r.text
    return r.json()["access_token"]

@pytest.fixture
def admin_headers(admin_token):
    return {"Authorization": f"Bearer {admin_token}"}

@pytest.fixture
def clinician_headers(clinician_token):
    return {"Authorization": f"Bearer {clinician_token}"}

@pytest.fixture
def sample_patient(client, admin_headers):
    r = client.post("/api/v1/patients", headers=admin_headers, json={
        "age": 48, "sex": "Female", "ethnicity": "South Asian",
        "family_history_cancer": 1, "family_history_diabetes": 1,
        "family_history_cardio": 0, "smoking_status": "never",
        "alcohol_units_weekly": 2.0, "exercise_min_weekly": 90})
    assert r.status_code == 201, r.text
    return r.json()

@pytest.fixture
def sample_checkup(client, admin_headers, sample_patient):
    r = client.post("/api/v1/checkups", headers=admin_headers, json={
        "patient_id": sample_patient["id"],
        "checkup_date": "2024-01-15",
        "hba1c": 6.1, "glucose_fasting": 113,
        "hemoglobin": 12.5, "lymphocytes_pct": 24,
        "wbc": 6.5, "cea": 3.8, "alt": 38,
        "ldl": 133, "hdl": 47,
        "bp_systolic": 139, "bmi": 28.6})
    assert r.status_code == 201, r.text
    return r.json()
