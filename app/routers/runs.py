from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.auth import DB, CurrentUser
from app.models import Failure, TestRun
from app.schemas import TestRunIn, TestRunOut
from app.services.triage import triage_service

router = APIRouter(prefix="/runs", tags=["test runs"])


@router.post("", response_model=TestRunOut, status_code=status.HTTP_201_CREATED)
def ingest_run(body: TestRunIn, db: DB, _: CurrentUser, auto_triage: bool = True):
    """Ingest one test run with its failures. New failures are auto-triaged when a model is loaded."""
    if body.passed and body.failures:
        raise HTTPException(status_code=422, detail="A passed run cannot carry failures")
    run = TestRun(**body.model_dump(exclude={"failures"}))
    run.failures = [Failure(**f.model_dump()) for f in body.failures]
    db.add(run)
    db.commit()
    triage_service.mark_dirty()

    if auto_triage and triage_service.model_ready:
        for failure in run.failures:
            triage_service.triage(db, failure)
    db.refresh(run)
    return run


@router.get("", response_model=list[TestRunOut])
def list_runs(
    db: DB,
    _: CurrentUser,
    build_id: str | None = None,
    scenario: str | None = None,
    passed: bool | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    stmt = select(TestRun).options(selectinload(TestRun.failures)).order_by(TestRun.started_at.desc())
    if build_id:
        stmt = stmt.where(TestRun.build_id == build_id)
    if scenario:
        stmt = stmt.where(TestRun.scenario == scenario)
    if passed is not None:
        stmt = stmt.where(TestRun.passed == passed)
    return db.scalars(stmt.limit(limit).offset(offset)).all()


@router.get("/{run_id}", response_model=TestRunOut)
def get_run(run_id: int, db: DB, _: CurrentUser):
    run = db.get(TestRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
