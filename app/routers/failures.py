from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.auth import DB, CurrentUser
from app.models import Failure
from app.schemas import FailureOut, FailureUpdate, TriageResult
from app.services.triage import ModelNotReady, triage_service

router = APIRouter(prefix="/failures", tags=["failures"])


def _get(db, failure_id: int) -> Failure:
    failure = db.get(Failure, failure_id)
    if not failure:
        raise HTTPException(status_code=404, detail="Failure not found")
    return failure


@router.get("", response_model=list[FailureOut])
def list_failures(
    db: DB,
    _: CurrentUser,
    status: str | None = None,
    subsystem: str | None = None,
    needs_review: bool = False,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    stmt = select(Failure).order_by(Failure.created_at.desc())
    if status:
        stmt = stmt.where(Failure.status == status)
    if subsystem:
        stmt = stmt.where(Failure.subsystem == subsystem)
    if needs_review:
        stmt = stmt.where(Failure.confidence < 0.6)
    return db.scalars(stmt.limit(limit).offset(offset)).all()


@router.get("/{failure_id}", response_model=FailureOut)
def get_failure(failure_id: int, db: DB, _: CurrentUser):
    return _get(db, failure_id)


@router.patch("/{failure_id}", response_model=FailureOut)
def update_failure(failure_id: int, body: FailureUpdate, db: DB, _: CurrentUser):
    """Human-in-the-loop: confirm or correct the label. Corrections become training data."""
    failure = _get(db, failure_id)
    for field, value in body.model_dump(exclude_none=True).items():
        setattr(failure, field, value)
    db.commit()
    return failure


@router.post("/{failure_id}/triage", response_model=TriageResult)
def triage_failure(failure_id: int, db: DB, _: CurrentUser):
    failure = _get(db, failure_id)
    try:
        return triage_service.triage(db, failure)
    except ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
