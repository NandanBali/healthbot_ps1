"""
Validated mental-health screeners: PHQ-9 (depression) and GAD-7 (anxiety).

Each item is answered on the standard 0–3 Likert scale:
  0 = Not at all
  1 = Several days
  2 = More than half the days
  3 = Nearly every day

Scores are summed and mapped to the standard severity bands. These are
widely-used screening instruments, not diagnostic tools — output always
carries a disclaimer.
"""

PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading or watching television",
    "Moving or speaking so slowly that other people could have noticed — or being fidgety/restless",
    "Thoughts that you would be better off dead, or of hurting yourself",
]

GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen",
]

LIKERT_LABELS = [
    "Not at all",
    "Several days",
    "More than half the days",
    "Nearly every day",
]

_DISCLAIMER = "(Screening tool only — not a diagnosis. Please consult a professional.)"


def _validate(answers, expected_len: int, name: str) -> list[int]:
    if len(answers) != expected_len:
        raise ValueError(
            f"{name} expects {expected_len} answers, got {len(answers)}."
        )
    coerced = []
    for a in answers:
        a = int(a)
        if not 0 <= a <= 3:
            raise ValueError(f"{name} answers must be in 0–3, got {a}.")
        coerced.append(a)
    return coerced


def _phq9_severity(score: int) -> str:
    if score >= 20:
        return "Severe depression"
    if score >= 15:
        return "Moderately severe depression"
    if score >= 10:
        return "Moderate depression"
    if score >= 5:
        return "Mild depression"
    return "Minimal depression"


def _gad7_severity(score: int) -> str:
    if score >= 15:
        return "Severe anxiety"
    if score >= 10:
        return "Moderate anxiety"
    if score >= 5:
        return "Mild anxiety"
    return "Minimal anxiety"


def score_phq9(answers: list) -> str:
    """Score a PHQ-9 questionnaire (9 items, each 0–3) and interpret it."""
    answers = _validate(answers, len(PHQ9_QUESTIONS), "PHQ-9")
    score = sum(answers)
    severity = _phq9_severity(score)

    # Item 9 flags self-harm thoughts — surface a safety note when endorsed.
    safety = ""
    if answers[8] >= 1:
        safety = (
            " ⚠️ You indicated thoughts of being better off dead or hurting "
            "yourself. Please reach out for support — US: 988, India: iCall "
            "+91 9152987821."
        )

    return (
        f"PHQ-9 score: {score}/27 — {severity}.{safety} {_DISCLAIMER}"
    )


def score_gad7(answers: list) -> str:
    """Score a GAD-7 questionnaire (7 items, each 0–3) and interpret it."""
    answers = _validate(answers, len(GAD7_QUESTIONS), "GAD-7")
    score = sum(answers)
    severity = _gad7_severity(score)
    return f"GAD-7 score: {score}/21 — {severity}. {_DISCLAIMER}"
