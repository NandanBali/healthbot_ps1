"""
Phase 4 — Programmatic cardiac risk scorer.

Point-based scoring (deterministic, no ML):
  age > 55        → +2
  systolic_bp > 140 → +3
  smokes          → +2

Tiers:
  0-2  → Low Risk
  3-4  → Moderate Risk
  5+   → High Risk
"""


def calculate_cardiac_risk(age: int, systolic_bp: int, smokes: bool) -> str:
    """Return a concise risk-tier summary string."""
    age = int(age)
    systolic_bp = int(systolic_bp)
    smokes = bool(smokes)

    score = 0
    factors = []

    if age > 55:
        score += 2
        factors.append(f"age {age} (>55, +2 pts)")
    if systolic_bp > 140:
        score += 3
        factors.append(f"BP {systolic_bp} mmHg (>140, +3 pts)")
    if smokes:
        score += 2
        factors.append("smoker (+2 pts)")

    if score >= 5:
        tier = "High Risk"
        advice = "Strongly recommend urgent cardiology consultation."
    elif score >= 3:
        tier = "Moderate Risk"
        advice = "Consider a full cardiac work-up with your GP."
    else:
        tier = "Low Risk"
        advice = "Maintain healthy lifestyle; routine check-ups advised."

    factor_str = ", ".join(factors) if factors else "no elevated risk factors"
    return (
        f"Cardiac Risk Score: {score}/7 — {tier}. "
        f"Factors: {factor_str}. {advice} "
        f"(Demo scoring model — not a clinical diagnosis.)"
    )
