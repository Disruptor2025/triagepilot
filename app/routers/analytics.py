from fastapi import APIRouter, HTTPException, Query

from app.auth import DB, AdminUser, CurrentUser
from app.services import analytics
from app.services.triage import triage_service

router = APIRouter(tags=["analytics"])


@router.get("/analytics/failures-by-subsystem")
def failures_by_subsystem(db: DB, _: CurrentUser):
    return analytics.failures_by_subsystem(db)


@router.get("/analytics/scenario-pass-rate")
def scenario_pass_rate(db: DB, _: CurrentUser, min_runs: int = Query(3, ge=1), limit: int = Query(20, ge=1, le=200)):
    return analytics.scenario_pass_rate(db, min_runs=min_runs, limit=limit)


@router.get("/analytics/build-regressions")
def build_regressions(db: DB, _: CurrentUser, threshold: float = -5.0, min_runs: int = Query(20, ge=1)):
    return analytics.build_regressions(db, threshold=threshold, min_runs=min_runs)


@router.get("/analytics/triage-accuracy")
def triage_accuracy(db: DB, _: CurrentUser):
    return analytics.triage_accuracy(db)


@router.post("/admin/retrain", tags=["admin"])
def retrain(db: DB, _: AdminUser):
    """Retrain the router on every human-labeled failure (including corrections)."""
    try:
        return triage_service.retrain(db)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
