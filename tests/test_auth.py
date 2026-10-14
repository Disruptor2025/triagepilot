def test_register_login_me(client):
    r = client.post("/auth/register", json={"email": "a@example.com", "password": "password123"})
    assert r.status_code == 201
    assert r.json()["role"] == "admin"  # first account bootstraps as admin

    token = client.post("/auth/token", data={"username": "a@example.com", "password": "password123"}).json()
    me = client.get("/auth/me", headers={"Authorization": f"Bearer {token['access_token']}"})
    assert me.json()["email"] == "a@example.com"


def test_second_user_is_engineer(engineer_headers, client):
    assert client.get("/auth/me", headers=engineer_headers).json()["role"] == "engineer"


def test_duplicate_email_rejected(client, admin_headers):
    r = client.post("/auth/register", json={"email": "admin@example.com", "password": "password123"})
    assert r.status_code == 409


def test_wrong_password(client, admin_headers):
    r = client.post("/auth/token", data={"username": "admin@example.com", "password": "nope-nope"})
    assert r.status_code == 401


def test_endpoints_require_token(client):
    assert client.get("/runs").status_code == 401
    assert client.get("/runs", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_retrain_is_admin_only(engineer_headers, client):
    assert client.post("/admin/retrain", headers=engineer_headers).status_code == 403
