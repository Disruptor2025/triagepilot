# TriagePilot

**AI-assisted triage for autonomous-vehicle test failures.**

AV programs run thousands of simulation and on-road tests per build. Every failure has to be
read, routed to the team that owns it (perception, prediction, planning, controls,
localization or infrastructure), checked against known issues, and tracked across builds.
At Zoox I did this by hand. TriagePilot automates the first pass:

- **Ingest** test runs and failures through a REST API (FastAPI + PostgreSQL).
- **Route** each failure to its owning subsystem with an ML classifier, with a confidence
  score; low-confidence calls go to a human review queue.
- **Retrieve** the most similar past failures and flag likely duplicates (the retrieval half of RAG).
- **Summarize** each failure as a triage note: by an LLM when an API key is set, grounded in the
  retrieved failures; by a deterministic template otherwise.
- **Analyze** with SQL: failure share by subsystem, worst scenarios, build-over-build pass-rate
  regressions (window functions), and how often the model agreed with humans.
- **Learn** from corrections: engineers relabel failures, admins retrain the model on them.

## Results

On a held-out split of synthetic failures (grouped by test run, so no leakage):

| Method | Accuracy | Macro F1 |
|---|---|---|
| Majority class | 0.319 | 0.081 |
| Rule: route by log module name | 0.764 | 0.740 |
| **TriagePilot router (TF-IDF + logistic regression)** | **0.911** | **0.900** |

With a 0.60 confidence threshold, **86% of failures are auto-routed at 97.6% accuracy**; the
remaining 14% go to human review. The analytics endpoint finds the injected regression in build
`2026.9.3` (pass rate down 15.7 points). Full report: `reports/evaluation.md`.

![confusion matrix](reports/confusion_matrix.png)

**About the data:** real fleet logs are proprietary, so `scripts/generate_data.py` produces
synthetic failures that encode real triage patterns: subsystem-specific vocabulary and log
modules, symptoms that are genuinely ambiguous (a hard brake can be perception, prediction or
planning), logs from a different module than the owner, and ~4% label noise.

## Architecture

```
            ┌────────────── FastAPI ──────────────┐
 client ──► │ /auth  /runs  /failures  /analytics │
            └──────┬──────────────┬───────────────┘
                   │              │
          SQLAlchemy ORM     TriageService
                   │          ├─ SubsystemRouter  (scikit-learn pipeline, joblib)
              PostgreSQL      ├─ FailureIndex     (TF-IDF cosine similarity)
            (SQLite in dev)   └─ summarize()      (Anthropic API, template fallback)
```

| Layer | Tech |
|---|---|
| API | FastAPI, Pydantic v2, OAuth2 password flow, JWT, bcrypt, role-based access |
| Data | PostgreSQL 16, SQLAlchemy 2.0, raw SQL analytics (CTEs, window functions) |
| ML | scikit-learn (word + character n-gram TF-IDF, logistic regression), pandas, matplotlib |
| LLM | Anthropic Messages API, optional, grounded on retrieved failures |
| Quality | pytest (21 tests, run on SQLite and PostgreSQL), ruff, GitHub Actions CI with a model-quality gate |
| Ops | Docker, docker-compose |

## Run it

```bash
# Local (SQLite)
pip install -r requirements-dev.txt
python -m scripts.generate_data          # 1,500 runs, 766 failures -> data/runs.json
python -m scripts.seed_db                # load + train the router
uvicorn app.main:app --reload            # http://localhost:8000/docs

# With PostgreSQL
docker compose up --build -d
docker compose exec api python -m scripts.seed_db

# Tests and evaluation
pytest -q
python -m scripts.evaluate               # writes reports/
```

The first account you register becomes admin. In the interactive docs, click **Authorize**,
sign in, then try `POST /runs`:

```json
{
  "build_id": "2026.10.3",
  "scenario": "unprotected_left_turn",
  "passed": false,
  "failures": [{
    "description": "Ego hesitated 14s at unprotected left turn, creeping behavior oscillated",
    "log_excerpt": "[88.1s] WARN planning/behavior: code=3121 frame=2210"
  }]
}
```

The failure comes back routed to `planning` (97% confidence) with a triage note and its nearest past failures.

## API

| Method | Path | What it does |
|---|---|---|
| POST | `/auth/register`, `/auth/token` | Create an account, get a JWT |
| POST | `/runs` | Ingest a run and its failures (auto-triaged) |
| GET | `/runs`, `/runs/{id}` | List with filters and pagination, get one |
| GET | `/failures?needs_review=true` | The human review queue |
| PATCH | `/failures/{id}` | Confirm or correct the label, change status |
| POST | `/failures/{id}/triage` | Re-run triage: owner, confidence, similar failures, duplicate, note |
| GET | `/analytics/failures-by-subsystem` | Share of failures and high-severity count per team |
| GET | `/analytics/scenario-pass-rate` | Worst scenarios first |
| GET | `/analytics/build-regressions` | Pass rate vs previous build, regressions flagged |
| GET | `/analytics/triage-accuracy` | Model precision per subsystem against human labels |
| POST | `/admin/retrain` | Retrain on all labeled failures (admin only) |


## Project layout

```
app/        FastAPI app: routers, models, schemas, auth, services, ml
scripts/    data generator, DB seeding, offline evaluation
tests/      API, analytics and ML tests
reports/    evaluation report and confusion matrix
```
