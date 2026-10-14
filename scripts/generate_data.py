"""Generate a synthetic but realistic AV test-failure dataset.

Real fleet logs are proprietary, so this generator encodes the failure patterns a test/triage
engineer sees day to day: each subsystem has its own vocabulary and log modules, and some
symptoms are genuinely ambiguous (a hard brake can be perception OR planning). Labels are
deliberately noisy (~4%) because human triage labels are noisy too.

Usage:  python -m scripts.generate_data --runs 1500 --out data/runs.json
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCENARIOS = [
    "unprotected_left_turn", "four_way_stop", "highway_merge", "school_zone", "construction_zone",
    "pedestrian_crosswalk", "cut_in_vehicle", "double_parked_car", "roundabout", "night_rain",
    "tunnel_entry", "emergency_vehicle", "pickup_dropoff", "cyclist_overtake", "lane_closure",
]
BUILDS = [f"2026.{m}.{p}" for m, p in [(8, 1), (8, 2), (8, 3), (9, 1), (9, 2), (9, 3), (10, 1), (10, 2)]]
REGRESSED_BUILD = "2026.9.3"  # a planning change that hurt pass rate; analytics should find it

AGENTS = ["pedestrian", "cyclist", "parked car", "truck", "scooter rider", "bus", "traffic cone", "plastic bag"]

TEMPLATES = {
    "perception": [
        "Missed detection of {agent} at {dist}m during {scenario}",
        "False positive obstacle: {agent} phantom track caused unnecessary slowdown",
        "Lidar point cloud dropout on {side} side for {ms}ms",
        "Camera exposure washout at tunnel exit, {agent} classified UNKNOWN",
        "Track ID swap between two {agent}s near occlusion",
        "Radar ghost return created stationary object in lane",
        "Classification flicker: {agent} toggling between classes for {ms}ms",
    ],
    "prediction": [
        "Predicted trajectory for {agent} did not anticipate turn, error {err}m at 3s",
        "Cut-in vehicle intent predicted too late during {scenario}",
        "Pedestrian jaywalking intent missed, yield initiated late",
        "Multimodal prediction collapsed to single mode for {agent}",
        "Occluded {agent} emerging from behind bus not forecasted",
    ],
    "planning": [
        "Ego hesitated {s}s at {scenario}, creeping behavior oscillated",
        "Planner selected lane change with insufficient gap to {agent}",
        "Hard brake {decel} m/s^2 triggered by conservative yield to {agent}",
        "Route replan loop: path toggled between two lanes",
        "Ego failed to nudge around double parked car, stuck {s}s",
        "Speed profile exceeded comfort jerk limit in {scenario}",
        "Ego blocked intersection after light turned red",
    ],
    "controls": [
        "Lateral tracking error {err}m exceeded threshold on curve",
        "Steering oscillation at {speed} mph, controller gain suspected",
        "Longitudinal overshoot of target speed by {speed} mph",
        "Brake actuator latency {ms}ms above spec",
        "Throttle command saturation during uphill start",
    ],
    "localization": [
        "Localization jump of {err}m in urban canyon, GNSS multipath",
        "Map mismatch: lane boundary offset {err}m from prior map",
        "Pose covariance spike after tunnel exit",
        "IMU drift during {s}s GNSS outage",
        "Ego localized in adjacent lane for {ms}ms",
    ],
    "infrastructure": [
        "Simulation timed out after {s}s, scenario did not complete",
        "Log upload failed, bag file corrupted at offset",
        "Node crashed with segfault in message bus",
        "CPU usage {pct}% caused dropped frames across pipeline",
        "Clock sync drift {ms}ms between compute units",
        "Docker image missing calibration file, run aborted",
    ],
}

# Ambiguous symptoms: the same description can belong to different owners.
AMBIGUOUS = [
    ("Unexpected hard brake for {agent} in {scenario}", ["perception", "planning", "prediction"]),
    ("Ego stopped short of stop line by {err}m", ["planning", "localization", "controls"]),
    ("Late yield to {agent} at crosswalk", ["prediction", "perception", "planning"]),
    ("Disengagement: safety driver took over in {scenario}", ["planning", "perception", "controls"]),
]

LOG_MODULES = {
    "perception": ["perception/lidar_seg", "perception/tracker", "perception/camera_det", "perception/fusion"],
    "prediction": ["prediction/agent_forecast", "prediction/intent_net"],
    "planning": ["planning/behavior", "planning/nav_planner", "planning/speed_opt"],
    "controls": ["controls/mpc_lat", "controls/long_ctrl", "vehicle/actuator_if"],
    "localization": ["localization/pose_filter", "localization/map_matcher", "sensors/gnss_imu"],
    "infrastructure": ["infra/sim_runner", "infra/log_uploader", "infra/msg_bus", "infra/scheduler"],
}
SEVERITY_WEIGHTS = {
    "perception": [0.10, 0.30, 0.40, 0.20], "prediction": [0.08, 0.32, 0.40, 0.20],
    "planning": [0.06, 0.30, 0.44, 0.20], "controls": [0.10, 0.30, 0.40, 0.20],
    "localization": [0.12, 0.33, 0.35, 0.20], "infrastructure": [0.00, 0.05, 0.35, 0.60],
}
SUBSYSTEM_MIX = {"perception": 0.27, "planning": 0.24, "prediction": 0.14, "controls": 0.10,
                 "localization": 0.10, "infrastructure": 0.15}


def _fill(t: str, rng: random.Random, scenario: str) -> str:
    return t.format(
        agent=rng.choice(AGENTS), dist=rng.randint(8, 90), scenario=scenario.replace("_", " "),
        side=rng.choice(["left", "right", "front", "rear"]), ms=rng.choice([80, 120, 250, 400, 900]),
        err=round(rng.uniform(0.2, 2.5), 1), s=rng.randint(2, 40), decel=round(rng.uniform(4.0, 8.5), 1),
        speed=rng.randint(3, 15), pct=rng.randint(91, 100),
    )


def _log(subsystem: str, rng: random.Random) -> str:
    mod = rng.choice(LOG_MODULES[subsystem])
    levels = ["WARN", "ERROR", "ERROR", "FATAL"] if subsystem == "infrastructure" else ["WARN", "ERROR"]
    level = rng.choice(levels)
    t = round(rng.uniform(1, 600), 2)
    return f"[{t}s] {level} {mod}: code={rng.randint(1000, 9999)} frame={rng.randint(1, 50000)}"


def make_failure(rng: random.Random, scenario: str, bias: str | None = None) -> dict:
    if rng.random() < 0.15:
        template, owners = rng.choice(AMBIGUOUS)
        subsystem = rng.choice(owners)
        description = _fill(template, rng, scenario)
    else:
        subsystem = bias if bias and rng.random() < 0.6 else rng.choices(
            list(SUBSYSTEM_MIX), weights=list(SUBSYSTEM_MIX.values()))[0]
        description = _fill(rng.choice(TEMPLATES[subsystem]), rng, scenario)

    # The log usually comes from the owning module, but not always (cross-module symptoms).
    log_owner = subsystem if rng.random() < 0.75 else rng.choice(list(LOG_MODULES))
    log_excerpt = "\n".join(_log(log_owner, rng) for _ in range(rng.randint(1, 3)))
    severity = rng.choices(["S0", "S1", "S2", "S3"], weights=SEVERITY_WEIGHTS[subsystem])[0]

    label = subsystem
    if rng.random() < 0.04:  # human label noise
        label = rng.choice([s for s in LOG_MODULES if s != subsystem])
    return {"description": description, "log_excerpt": log_excerpt, "subsystem": label, "severity": severity}


def generate(n_runs: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    start = datetime(2026, 6, 1, tzinfo=UTC)
    runs = []
    for i in range(n_runs):
        build = BUILDS[min(i * len(BUILDS) // n_runs, len(BUILDS) - 1)]
        scenario = rng.choice(SCENARIOS)
        base_fail = 0.30 + (0.18 if build == REGRESSED_BUILD else 0)
        base_fail += 0.10 if scenario in {"night_rain", "construction_zone"} else 0
        passed = rng.random() > base_fail
        bias = "planning" if build == REGRESSED_BUILD else None
        failures = [] if passed else [make_failure(rng, scenario, bias) for _ in range(rng.choice([1, 1, 1, 2, 3]))]
        runs.append({
            "build_id": build,
            "scenario": scenario,
            "source": "road" if rng.random() < 0.2 else "sim",
            "passed": passed,
            "duration_s": round(rng.uniform(20, 600), 1),
            "started_at": (start + timedelta(minutes=90 * i)).isoformat(),
            "failures": failures,
        })
    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=1500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="data/runs.json")
    a = p.parse_args()
    runs = generate(a.runs, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(runs, indent=1))
    n_fail = sum(len(r["failures"]) for r in runs)
    print(f"Wrote {len(runs)} runs with {n_fail} failures to {a.out}")


if __name__ == "__main__":
    main()
