"""Triage summary writer.

If ANTHROPIC_API_KEY is set, an LLM writes the summary, grounded in the failure text and the
retrieved similar failures (retrieval-augmented generation). Without a key, or if the call
fails, a deterministic template is used, so the service and its tests never depend on a network.
"""
from __future__ import annotations

import logging

import httpx

from app.config import get_settings

log = logging.getLogger(__name__)

PROMPT = """You are an autonomous-vehicle test triage engineer. Write a triage note of at most 4 sentences:
what failed, the most likely owning subsystem and why, and whether it matches a known issue.
Use only the information below. Do not invent log lines.

New failure:
{failure}

Model prediction: {label} (confidence {confidence:.0%})

Most similar past failures:
{similar}
"""


def _template(failure_text: str, label: str, confidence: float, similar: list[dict]) -> str:
    first_line = failure_text.splitlines()[0][:160]
    note = f"Failure: {first_line}. Routed to {label} ({confidence:.0%} confidence)."
    if similar:
        top = similar[0]
        note += f" Closest past failure is #{top['failure_id']} (similarity {top['score']:.2f}"
        note += f", owned by {top['subsystem']})." if top.get("subsystem") else ")."
    if confidence < 0.6:
        note += " Low confidence: needs human review before assignment."
    return note


def summarize(failure_text: str, label: str, confidence: float, similar: list[dict]) -> str:
    settings = get_settings()
    if not settings.anthropic_api_key:
        return _template(failure_text, label, confidence, similar)

    similar_txt = "\n".join(
        f"- #{s['failure_id']} [{s.get('subsystem') or 'unlabeled'}] (sim {s['score']:.2f}): {s['description']}"
        for s in similar
    ) or "- none"
    try:
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "max_tokens": 300,
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT.format(
                            failure=failure_text, label=label, confidence=confidence, similar=similar_txt
                        ),
                    }
                ],
            },
            timeout=20,
        )
        resp.raise_for_status()
        return "".join(b.get("text", "") for b in resp.json()["content"]).strip()
    except Exception as exc:  # network, auth, quota: fall back rather than fail triage
        log.warning("LLM summary failed, using template: %s", exc)
        return _template(failure_text, label, confidence, similar)
