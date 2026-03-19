"""Tests for authentication endpoints."""
import pytest


class TestRegister:
    def test_register_success(self, client):
        r = client.post("/api/v1/auth/register", json={
            "username": "newuser", "email": "new@test.com",
            "password": "password123", "role": "clinician"
        })
        assert r.status_code == 201
        data = r.json()
        assert "access_token" in data
        assert data["user"]["username"] == "newuser"
        assert data["user"]["role"] == "clinician"

    def test_register_duplicate_username(self, client):
        payload = {"username": "dup", "email": "dup@test.com",
                   "password": "pass1234", "role": "clinician"}
        client.post("/api/v1/auth/register", json=payload)
        r = client.post("/api/v1/auth/register", json=payload)
        assert r.status_code == 400
        assert "taken" in r.json()["detail"].lower()

    def test_register_different_roles(self, client):
        for role in ["admin", "clinician", "researcher"]:
            r = client.post("/api/v1/auth/register", json={
                "username": f"user_{role}", "email": f"{role}@test.com",
                "password": "pass1234", "role": role
            })
            assert r.status_code == 201
            assert r.json()["user"]["role"] == role


class TestLogin:
    def test_login_success(self, client):
        client.post("/api/v1/auth/register", json={
            "username": "logintest", "email": "login@test.com",
            "password": "mypassword", "role": "clinician"
        })
        r = client.post("/api/v1/auth/login", json={
            "username": "logintest", "password": "mypassword"
        })
        assert r.status_code == 200
        assert "access_token" in r.json()
        assert r.json()["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        client.post("/api/v1/auth/register", json={
            "username": "wrongpw", "email": "wp@test.com",
            "password": "correct", "role": "clinician"
        })
        r = client.post("/api/v1/auth/login", json={
            "username": "wrongpw", "password": "wrong"
        })
        assert r.status_code == 401

    def test_login_nonexistent_user(self, client):
        r = client.post("/api/v1/auth/login", json={
            "username": "ghost", "password": "pass"
        })
        assert r.status_code == 401

    def test_me_endpoint(self, client, admin_headers):
        r = client.get("/api/v1/auth/me", headers=admin_headers)
        assert r.status_code == 200
        data = r.json()
        assert data["username"] == "admin"
        assert "id" in data
        assert "hashed_password" not in data   # never expose password hash

    def test_unauthenticated_request(self, client):
        r = client.get("/api/v1/patients")
        assert r.status_code in (401, 403)  # no token → unauthorized

    def test_invalid_token(self, client):
        r = client.get("/api/v1/patients", headers={
            "Authorization": "Bearer this-is-not-a-valid-token"
        })
        assert r.status_code == 401


class TestPasswordChange:
    def test_change_password(self, client, clinician_headers, clinician_token):
        r = client.put("/api/v1/auth/password", headers=clinician_headers, json={
            "current_password": "doctorpass123",
            "new_password": "newstrongpass456"
        })
        assert r.status_code == 200
        assert r.json()["message"] == "Password changed successfully"
        # Verify old password no longer works
        r2 = client.post("/api/v1/auth/login", json={
            "username": "dr_test", "password": "doctorpass123"
        })
        assert r2.status_code == 401
        # Verify new password works
        r3 = client.post("/api/v1/auth/login", json={
            "username": "dr_test", "password": "newstrongpass456"
        })
        assert r3.status_code == 200

    def test_change_password_wrong_current(self, client, clinician_headers):
        r = client.put("/api/v1/auth/password", headers=clinician_headers, json={
            "current_password": "wrongcurrent",
            "new_password": "newpass789"
        })
        assert r.status_code == 400

    def test_change_password_too_short(self, client, clinician_headers):
        r = client.put("/api/v1/auth/password", headers=clinician_headers, json={
            "current_password": "doctorpass123",
            "new_password": "short"
        })
        assert r.status_code == 400
