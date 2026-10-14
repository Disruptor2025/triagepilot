from app.ml.retrieval import FailureIndex
from app.ml.router_model import SubsystemRouter
from app.ml.summarizer import summarize
from scripts.generate_data import generate


def _data():
    rows = [f for r in generate(600, seed=11) for f in r["failures"]]
    return [f"{f['description']}\n{f['log_excerpt']}" for f in rows], [f["subsystem"] for f in rows]


def test_router_learns_and_round_trips(tmp_path):
    texts, labels = _data()
    model = SubsystemRouter.train(texts, labels)
    pred = model.predict("Localization jump of 2.1m in urban canyon, GNSS multipath")
    assert pred.label == "localization"
    assert abs(sum(p for _, p in model.predict("x" * 20, top_k=6).ranked) - 1) < 1e-6

    path = tmp_path / "m.joblib"
    model.save(path)
    assert SubsystemRouter.load(path).predict_many(texts[:20]) == model.predict_many(texts[:20])


def test_index_excludes_self_and_ranks_exact_match_first():
    idx = FailureIndex()
    idx.index([1, 2, 3], ["lidar dropout left side", "steering oscillation at 10 mph", "lidar dropout right side"])
    hits = idx.search("lidar dropout left side", k=3)
    assert hits[0].doc_id == 1 and hits[0].score > 0.99
    assert all(h.doc_id != 1 for h in idx.search("lidar dropout left side", exclude_id=1))
    assert FailureIndex().search("anything") == []


def test_template_summary_flags_low_confidence():
    s = summarize("Ego hesitated 12s at four way stop", "planning", 0.41,
                  [{"failure_id": 7, "score": 0.66, "subsystem": "planning", "description": "x"}])
    assert "planning" in s and "#7" in s and "human review" in s
