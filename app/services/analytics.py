"""Analytics written as plain SQL (runs on both PostgreSQL and SQLite 3.25+).

These are the questions a test-ops lead asks every morning: where are we failing,
which builds regressed, and can we trust the auto-triage?
"""
from sqlalchemy import text
from sqlalchemy.orm import Session

FAILURES_BY_SUBSYSTEM = text(
    """
    SELECT COALESCE(f.subsystem, 'unlabeled') AS subsystem,
           COUNT(*) AS failures,
           SUM(CASE WHEN f.severity IN ('S0', 'S1') THEN 1 ELSE 0 END) AS high_severity,
           ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_all
    FROM failures f
    GROUP BY COALESCE(f.subsystem, 'unlabeled')
    ORDER BY failures DESC
    """
)

SCENARIO_PASS_RATE = text(
    """
    SELECT r.scenario,
           COUNT(*) AS runs,
           SUM(CASE WHEN r.passed THEN 1 ELSE 0 END) AS passed,
           ROUND(100.0 * SUM(CASE WHEN r.passed THEN 1 ELSE 0 END) / COUNT(*), 1) AS pass_rate
    FROM test_runs r
    GROUP BY r.scenario
    HAVING COUNT(*) >= :min_runs
    ORDER BY pass_rate ASC, runs DESC
    LIMIT :limit
    """
)

# Pass rate per build, compared with the previous build (LAG window function) to flag regressions.
BUILD_REGRESSIONS = text(
    """
    WITH per_build AS (
        SELECT r.build_id,
               MIN(r.started_at) AS first_seen,
               COUNT(*) AS runs,
               ROUND(100.0 * SUM(CASE WHEN r.passed THEN 1 ELSE 0 END) / COUNT(*), 1) AS pass_rate
        FROM test_runs r
        GROUP BY r.build_id
        HAVING COUNT(*) >= :min_runs  -- tiny samples are noise, not regressions
    )
    SELECT build_id, runs, pass_rate,
           LAG(pass_rate) OVER (ORDER BY first_seen) AS prev_pass_rate,
           ROUND(pass_rate - LAG(pass_rate) OVER (ORDER BY first_seen), 1) AS delta
    FROM per_build
    ORDER BY first_seen
    """
)

# How often the model agreed with the human label, per subsystem (precision of the router).
TRIAGE_ACCURACY = text(
    """
    SELECT f.predicted_subsystem AS predicted,
           COUNT(*) AS predicted_count,
           SUM(CASE WHEN f.predicted_subsystem = f.subsystem THEN 1 ELSE 0 END) AS correct,
           ROUND(100.0 * SUM(CASE WHEN f.predicted_subsystem = f.subsystem THEN 1 ELSE 0 END) / COUNT(*), 1)
               AS precision_pct,
           ROUND(CAST(AVG(f.confidence) AS NUMERIC), 3) AS avg_confidence
    FROM failures f
    WHERE f.predicted_subsystem IS NOT NULL AND f.subsystem IS NOT NULL
    GROUP BY f.predicted_subsystem
    ORDER BY precision_pct ASC
    """
)


def _rows(db: Session, stmt, **params) -> list[dict]:
    return [dict(r._mapping) for r in db.execute(stmt, params)]


def failures_by_subsystem(db: Session) -> list[dict]:
    return _rows(db, FAILURES_BY_SUBSYSTEM)


def scenario_pass_rate(db: Session, min_runs: int = 3, limit: int = 20) -> list[dict]:
    return _rows(db, SCENARIO_PASS_RATE, min_runs=min_runs, limit=limit)


def build_regressions(db: Session, threshold: float = -5.0, min_runs: int = 20) -> list[dict]:
    rows = _rows(db, BUILD_REGRESSIONS, min_runs=min_runs)
    for r in rows:
        r["regression"] = r["delta"] is not None and r["delta"] <= threshold
    return rows


def triage_accuracy(db: Session) -> list[dict]:
    return _rows(db, TRIAGE_ACCURACY)
