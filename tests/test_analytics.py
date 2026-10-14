def test_failures_by_subsystem_percentages_sum_to_100(seeded, client):
    rows = client.get("/analytics/failures-by-subsystem", headers=seeded).json()
    assert {r["subsystem"] for r in rows} >= {"perception", "planning"}
    assert abs(sum(r["pct_of_all"] for r in rows) - 100) < 1
    assert rows == sorted(rows, key=lambda r: r["failures"], reverse=True)


def test_scenario_pass_rate_sorted_worst_first(seeded, client):
    rows = client.get("/analytics/scenario-pass-rate?min_runs=5", headers=seeded).json()
    assert rows and all(r["runs"] >= 5 for r in rows)
    rates = [r["pass_rate"] for r in rows]
    assert rates == sorted(rates)


def test_build_regression_detected(seeded, client):
    rows = client.get("/analytics/build-regressions", headers=seeded).json()
    assert rows[0]["prev_pass_rate"] is None  # first build has nothing to compare against
    flagged = [r["build_id"] for r in rows if r["regression"]]
    assert "2026.9.3" in flagged  # the generator injects a regression in this build


def test_triage_accuracy_after_auto_triage(seeded, client):
    for f in client.get("/failures?limit=40", headers=seeded).json():
        client.post(f"/failures/{f['id']}/triage", headers=seeded)
    rows = client.get("/analytics/triage-accuracy", headers=seeded).json()
    assert rows
    total = sum(r["predicted_count"] for r in rows)
    correct = sum(r["correct"] for r in rows)
    assert correct / total > 0.75


def test_small_builds_are_not_flagged(seeded, client):
    run = {"build_id": "tiny-build", "scenario": "roundabout", "passed": False,
           "failures": [{"description": "Node crashed with segfault in message bus"}]}
    client.post("/runs", json=run, headers=seeded)
    builds = [r["build_id"] for r in client.get("/analytics/build-regressions", headers=seeded).json()]
    assert "tiny-build" not in builds
