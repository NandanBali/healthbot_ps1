"""
Mental-health crisis guardrail — deterministic self-harm/suicide intercept.

Mirrors ``emergency_triage.py``: this must be evaluated BEFORE any LLM call.
It uses plain string matching so it cannot be confused or hallucinated by a
model. If any trigger is present, the app surfaces crisis-hotline information
and bypasses the LLM entirely.
"""

CRISIS_TRIGGERS = [
    "kill myself",
    "killing myself",
    "want to die",
    "wanna die",
    "end my life",
    "ending my life",
    "suicidal",
    "suicide",
    "self harm",
    "self-harm",
    "hurt myself",
    "harming myself",
    "no reason to live",
    "don't want to be alive",
    "do not want to be alive",
    "better off dead",
    "take my own life",
]

CRISIS_MESSAGE = (
    "🆘 You're not alone, and help is available right now. "
    "If you are thinking about harming yourself, please reach out immediately:\n"
    "- US: call or text 988 (Suicide & Crisis Lifeline)\n"
    "- India: iCall +91 9152987821, AASRA +91 9820466726\n"
    "- Or call your local emergency number (911 / 112 / 102).\n"
    "If you are in immediate danger, please contact emergency services now. "
    "Talking to a trusted person or a mental-health professional can help."
)


def check_crisis_triggers(user_text: str) -> bool:
    """Return True if user_text contains any self-harm / suicidal phrase."""
    lowered = user_text.lower()
    return any(trigger in lowered for trigger in CRISIS_TRIGGERS)
