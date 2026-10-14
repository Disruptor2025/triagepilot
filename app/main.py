"""TriagePilot API: AI-assisted triage for autonomous-vehicle test failures."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings
from app.database import Base, engine
from app.routers import analytics, auth, failures, runs
from app.services.triage import triage_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    Base.metadata.create_all(bind=engine)
    triage_service.load_model()
    yield


app = FastAPI(
    title=get_settings().app_name,
    description="Ingest AV test runs, auto-route failures to the owning subsystem, "
    "surface similar past failures, and track pass-rate regressions across builds.",
    version="1.0.0",
    lifespan=lifespan,
)
app.include_router(auth.router)
app.include_router(runs.router)
app.include_router(failures.router)
app.include_router(analytics.router)


@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "model_loaded": triage_service.model_ready}
