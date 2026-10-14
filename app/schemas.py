"""Pydantic request/response schemas."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field

Subsystem = Literal["perception", "prediction", "planning", "controls", "localization", "infrastructure"]
Severity = Literal["S0", "S1", "S2", "S3"]


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    email: EmailStr
    role: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class FailureIn(BaseModel):
    description: str = Field(min_length=5, examples=["Ego hard-braked for a plastic bag crossing the lane"])
    log_excerpt: str = Field(default="", examples=["[perception] obj_id=4411 class=UNKNOWN conf=0.41"])
    subsystem: Subsystem | None = None
    severity: Severity | None = None


class TestRunIn(BaseModel):
    build_id: str = Field(examples=["2026.10.2-rc3"])
    scenario: str = Field(examples=["unprotected_left_turn"])
    source: Literal["sim", "road"] = "sim"
    passed: bool = True
    duration_s: float = Field(default=0.0, ge=0)
    failures: list[FailureIn] = []


class FailureOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    run_id: int
    description: str
    log_excerpt: str
    subsystem: str | None
    severity: str | None
    predicted_subsystem: str | None
    confidence: float | None
    duplicate_of: int | None
    summary: str | None
    status: str
    created_at: datetime


class TestRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    build_id: str
    scenario: str
    source: str
    passed: bool
    duration_s: float
    started_at: datetime
    failures: list[FailureOut] = []


class FailureUpdate(BaseModel):
    """A human correcting or confirming triage. These labels feed the next retrain."""
    subsystem: Subsystem | None = None
    severity: Severity | None = None
    status: Literal["new", "triaged", "resolved"] | None = None


class SimilarFailure(BaseModel):
    failure_id: int
    score: float
    description: str
    subsystem: str | None


class TriageResult(BaseModel):
    failure_id: int
    predicted_subsystem: str
    confidence: float
    top_candidates: list[tuple[str, float]]
    similar: list[SimilarFailure]
    likely_duplicate_of: int | None
    summary: str
    needs_human_review: bool
