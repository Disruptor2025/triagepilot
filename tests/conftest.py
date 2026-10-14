import os
import tempfile

# Point the app at a throwaway database and model path BEFORE it is imported.
_tmp = tempfile.mkdtemp()
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp}/test.db")
os.environ["MODEL_PATH"] = f"{_tmp}/router.joblib"
os.environ["SECRET_KEY"] = "test-secret-key-that-is-at-least-32-bytes"
os.environ.pop("ANTHROPIC_API_KEY", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.services.triage import triage_service  # noqa: E402
from scripts.generate_data import generate  # noqa: E402

get_settings.cache_clear()


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    if os.path.exists(os.environ["MODEL_PATH"]):  # each test starts with no trained model
        os.remove(os.environ["MODEL_PATH"])
    triage_service._router = None
    triage_service.mark_dirty()
    with TestClient(app) as c:
        yield c


def _auth(client, email, password="password123"):
    client.post("/auth/register", json={"email": email, "password": password})
    token = client.post("/auth/token", data={"username": email, "password": password}).json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def admin_headers(client):
    return _auth(client, "admin@example.com")  # first user becomes admin


@pytest.fixture()
def engineer_headers(client, admin_headers):
    return _auth(client, "eng@example.com")


@pytest.fixture()
def seeded(client, admin_headers):
    """Load a small synthetic history (without auto-triage) and train the model through the API."""
    for run in generate(400, seed=3):
        run.pop("started_at")
        r = client.post("/runs?auto_triage=false", json=run, headers=admin_headers)
        assert r.status_code == 201, r.text
    r = client.post("/admin/retrain", headers=admin_headers)
    assert r.status_code == 200, r.text
    return admin_headers
