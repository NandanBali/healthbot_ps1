"""
Phase 2 — High-risk deterministic guardrails.

This module must be evaluated BEFORE any LLM call.  It uses plain string
matching so it cannot be confused or hallucinated by a model.
"""

EMERGENCY_TRIGGERS = [
    "chest pain",
    "crushing pressure",
    "left arm pain",
    "heart attack",
    "can't breathe",
    "cannot breathe",
    "shortness of breath",
    "cardiac arrest",
    "jaw pain",
    "crushing chest",
]

EMERGENCY_MESSAGE = (
    "🚨 EMERGENCY LOGGED: Please call emergency services (911/102) immediately."
)


def check_emergency_triggers(user_text: str) -> bool:
    """Return True if user_text contains any acute high-risk cardiac phrase."""
    lowered = user_text.lower()
    return any(trigger in lowered for trigger in EMERGENCY_TRIGGERS)
