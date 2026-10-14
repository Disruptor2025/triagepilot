"""Load generated runs into the database and train the first router model.

Usage:  python -m scripts.seed_db --data data/runs.json
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime

from sqlalchemy import func, select

from app.database import Base, SessionLocal, engine
from app.models import Failure, TestRun
from app.services.triage import triage_service


def seed(path: str) -> None:
    Base.metadata.create_all(bind=engine)
    runs = json.loads(open(path).read())
    with SessionLocal() as db:
        if db.scalar(select(func.count(TestRun.id))):
            print("Database already has runs; skipping load.")
        else:
            for r in runs:
                run = TestRun(**{k: v for k, v in r.items() if k not in {"failures", "started_at"}},
                              started_at=datetime.fromisoformat(r["started_at"]))
                run.failures = [Failure(**f) for f in r["failures"]]
                db.add(run)
            db.commit()
            print(f"Loaded {len(runs)} runs.")
        print("Retrained router:", triage_service.retrain(db))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/runs.json")
    seed(p.parse_args().data)
