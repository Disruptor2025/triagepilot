"""Triage service: ties the router model, the similarity index and the summarizer together."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ml.retrieval import DUPLICATE_THRESHOLD, FailureIndex
from app.ml.router_model import SubsystemRouter
from app.ml.summarizer import summarize
from app.models import Failure
from app.schemas import SimilarFailure, TriageResult

log = logging.getLogger(__name__)


class ModelNotReady(RuntimeError):
    pass


class TriageService:
    def __init__(self) -> None:
        self._router: SubsystemRouter | None = None
        self._index = FailureIndex()
        self._index_dirty = True
        self._lock = threading.Lock()

    # ---- model lifecycle -------------------------------------------------
    def load_model(self, path: str | None = None) -> bool:
        path = path or get_settings().model_path
        if Path(path).exists():
            self._router = SubsystemRouter.load(path)
            log.info("Loaded router model from %s", path)
            return True
        log.warning("No router model at %s; POST /admin/retrain to train one", path)
        return False

    @property
    def model_ready(self) -> bool:
        return self._router is not None

    def retrain(self, db: Session) -> dict:
        rows = db.scalars(select(Failure).where(Failure.subsystem.is_not(None))).all()
        labels = {r.subsystem for r in rows}
        if len(rows) < 30 or len(labels) < 2:
            raise ValueError(f"Need >=30 labeled failures across >=2 subsystems, have {len(rows)}")
        router = SubsystemRouter.train([r.text for r in rows], [r.subsystem for r in rows])
        router.save(get_settings().model_path)
        self._router = router
        return {"trained_on": len(rows), "classes": sorted(labels)}

    def mark_dirty(self) -> None:
        self._index_dirty = True

    def _refresh_index(self, db: Session) -> None:
        with self._lock:
            if not self._index_dirty:
                return
            rows = db.execute(select(Failure.id, Failure.description, Failure.log_excerpt)).all()
            self._index.index([r.id for r in rows], [f"{r.description}\n{r.log_excerpt}" for r in rows])
            self._index_dirty = False

    # ---- triage ----------------------------------------------------------
    def triage(self, db: Session, failure: Failure, k: int = 3) -> TriageResult:
        if self._router is None:
            raise ModelNotReady("Router model not trained yet")
        self._refresh_index(db)

        pred = self._router.predict(failure.text)
        hits = self._index.search(failure.text, k=k, exclude_id=failure.id)
        by_id = {f.id: f for f in db.scalars(select(Failure).where(Failure.id.in_([h.doc_id for h in hits])))}
        similar = [
            SimilarFailure(
                failure_id=h.doc_id,
                score=h.score,
                description=by_id[h.doc_id].description,
                subsystem=by_id[h.doc_id].subsystem,
            )
            for h in hits
            if h.doc_id in by_id
        ]
        duplicate = similar[0].failure_id if similar and similar[0].score >= DUPLICATE_THRESHOLD else None
        summary = summarize(failure.text, pred.label, pred.confidence, [s.model_dump() for s in similar])

        failure.predicted_subsystem = pred.label
        failure.confidence = pred.confidence
        failure.duplicate_of = duplicate
        failure.summary = summary
        if failure.status == "new":
            failure.status = "triaged"
        db.commit()

        return TriageResult(
            failure_id=failure.id,
            predicted_subsystem=pred.label,
            confidence=pred.confidence,
            top_candidates=pred.ranked,
            similar=similar,
            likely_duplicate_of=duplicate,
            summary=summary,
            needs_human_review=pred.needs_review,
        )


triage_service = TriageService()
