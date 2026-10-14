"""ORM models.

A TestRun is one execution of a scenario (on-road drive or simulation) against a software build.
A Failure is an issue observed during a run. Triage = deciding which team owns a failure,
how severe it is, and whether it looks like a known issue.
"""
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

SUBSYSTEMS = ("perception", "prediction", "planning", "controls", "localization", "infrastructure")
SEVERITIES = ("S0", "S1", "S2", "S3")  # S0 = safety-critical ... S3 = cosmetic


def _now() -> datetime:
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="engineer")  # engineer | admin
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class TestRun(Base):
    __tablename__ = "test_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    build_id: Mapped[str] = mapped_column(String(50), index=True)
    scenario: Mapped[str] = mapped_column(String(100), index=True)
    source: Mapped[str] = mapped_column(String(20), default="sim")  # sim | road
    passed: Mapped[bool] = mapped_column(Boolean, default=True)
    duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)

    failures: Mapped[list["Failure"]] = relationship(back_populates="run", cascade="all, delete-orphan")


class Failure(Base):
    __tablename__ = "failures"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("test_runs.id", ondelete="CASCADE"), index=True)
    description: Mapped[str] = mapped_column(Text)
    log_excerpt: Mapped[str] = mapped_column(Text, default="")

    # Ground truth, filled in by a human (or by historical data).
    subsystem: Mapped[str | None] = mapped_column(String(30), nullable=True, index=True)
    severity: Mapped[str | None] = mapped_column(String(5), nullable=True)

    # Model output.
    predicted_subsystem: Mapped[str | None] = mapped_column(String(30), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    duplicate_of: Mapped[int | None] = mapped_column(Integer, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="new")  # new | triaged | resolved
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    run: Mapped[TestRun] = relationship(back_populates="failures")

    @property
    def text(self) -> str:
        """The text the models read: what a triage engineer would read first."""
        return f"{self.description}\n{self.log_excerpt}".strip()
