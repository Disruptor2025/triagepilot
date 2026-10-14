RUN = {
    "build_id": "2026.10.2",
    "scenario": "night_rain",
    "source": "road",
    "passed": False,
    "duration_s": 120,
    "failures": [
        {
            "description": "Lidar point cloud dropout on left side for 400ms",
            "log_excerpt": "[12.5s] ERROR perception/lidar_seg: code=4411 frame=900",
        }
    ],
}


def test_ingest_without_model_still_stores_run(client, admin_headers):
    r = client.post("/runs", json=RUN, headers=admin_headers)
    assert r.status_code == 201
    body = r.json()
    assert body["failures"][0]["status"] == "new"
    assert body["failures"][0]["predicted_subsystem"] is None


def test_triage_without_model_returns_503(client, admin_headers):
    fid = client.post("/runs", json=RUN, headers=admin_headers).json()["failures"][0]["id"]
    assert client.post(f"/failures/{fid}/triage", headers=admin_headers).status_code == 503


def test_passed_run_cannot_have_failures(client, admin_headers):
    bad = {**RUN, "passed": True}
    assert client.post("/runs", json=bad, headers=admin_headers).status_code == 422


def test_auto_triage_routes_to_owner(seeded, client):
    r = client.post("/runs", json=RUN, headers=seeded)
    f = r.json()["failures"][0]
    assert f["predicted_subsystem"] == "perception"
    assert f["status"] == "triaged"
    assert f["summary"]


def test_triage_returns_similar_and_duplicate(seeded, client):
    fid = client.post("/runs?auto_triage=false", json=RUN, headers=seeded).json()["failures"][0]["id"]
    dup_id = client.post("/runs?auto_triage=false", json=RUN, headers=seeded).json()["failures"][0]["id"]
    result = client.post(f"/failures/{dup_id}/triage", headers=seeded).json()

    assert result["predicted_subsystem"] == "perception"
    assert 0 < result["confidence"] <= 1
    assert len(result["top_candidates"]) == 3
    assert result["similar"], "expected retrieved neighbours"
    assert all(s["failure_id"] != dup_id for s in result["similar"])
    assert result["likely_duplicate_of"] == fid


def test_human_correction_and_review_filter(seeded, client):
    fid = client.post("/runs", json=RUN, headers=seeded).json()["failures"][0]["id"]
    r = client.patch(f"/failures/{fid}", json={"subsystem": "localization", "status": "resolved"}, headers=seeded)
    assert r.json()["subsystem"] == "localization"
    assert r.json()["status"] == "resolved"
    assert client.patch(f"/failures/{fid}", json={"subsystem": "weather"}, headers=seeded).status_code == 422


def test_list_runs_filters_and_pagination(seeded, client):
    rows = client.get("/runs?passed=false&limit=5", headers=seeded).json()
    assert len(rows) == 5
    assert all(not r["passed"] for r in rows)
    assert client.get("/runs/999999", headers=seeded).status_code == 404
