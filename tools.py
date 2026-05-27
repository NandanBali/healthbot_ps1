"""
Core tool functions for the cardiac health PoC.

These are bound to the LLM via OpenAI function calling, so each function
must return a plain string the model can read as a tool result.
"""

EMERGENCY_KEYWORDS = [
    "chest pain",
    "heart attack",
    "shortness of breath",
    "can't breathe",
    "cannot breathe",
    "arm pain",
    "jaw pain",
    "cardiac arrest",
    "crushing chest",
]


def check_emergency(user_text: str) -> bool:
    """Return True if the text contains any emergency cardiac keyword."""
    lowered = user_text.lower()
    return any(kw in lowered for kw in EMERGENCY_KEYWORDS)


def calculate_heart_risk(age: int, systolic_bp: int, smokes: bool) -> str:
    """
    Very basic rule-based cardiac risk estimate.

    Rules (simplified Framingham-style tiers):
      - High:   age > 50 AND bp > 140, OR smokes AND (age > 45 OR bp > 130)
      - Medium: age > 40 OR bp > 130
      - Low:    everything else
    """
    age = int(age)
    systolic_bp = int(systolic_bp)
    smokes = bool(smokes)

    if (age > 50 and systolic_bp > 140) or (smokes and (age > 45 or systolic_bp > 130)):
        risk = "High"
        advice = "Please consult a cardiologist soon."
    elif age > 40 or systolic_bp > 130:
        risk = "Medium"
        advice = "Consider a check-up and lifestyle adjustments."
    else:
        risk = "Low"
        advice = "Keep up healthy habits."

    smoking_note = "smoker" if smokes else "non-smoker"
    return (
        f"Risk assessment — Age: {age}, BP: {systolic_bp} mmHg, {smoking_note}. "
        f"Estimated risk: **{risk}**. {advice} "
        f"(This is a simplified demo, not medical advice.)"
    )


def analyze_ecg_data(signal_array: list) -> str:
    """
    Simulate ECG analysis using basic statistics on a numeric array.

    Interprets inter-peak-style data: values are treated as BPM samples or
    raw amplitude readings.  A simple threshold + average determines status.
    """
    if not signal_array:
        return "No ECG data provided."

    nums = [float(x) for x in signal_array]
    average = sum(nums) / len(nums)
    peaks_above = sum(1 for v in nums if v > 5)
    variability = max(nums) - min(nums)

    # Derive a rough BPM estimate: average of readings scaled to [40, 180]
    # (pure demo heuristic — not clinically meaningful)
    scaled_bpm = min(max(int(average * 10), 40), 180)

    if 60 <= scaled_bpm <= 100 and variability < 15:
        status = f"Average pulse looks normal ({scaled_bpm} BPM). Rhythm appears regular."
    elif scaled_bpm < 60:
        status = f"Pulse looks low ({scaled_bpm} BPM). Possible bradycardia — worth checking."
    elif scaled_bpm > 100:
        status = f"Pulse looks elevated ({scaled_bpm} BPM). Possible tachycardia — worth checking."
    else:
        status = f"Pulse looks irregular ({scaled_bpm} BPM, variability {variability:.1f})."

    return (
        f"ECG analysis — {len(nums)} samples, avg={average:.2f}, "
        f"peaks above threshold: {peaks_above}. {status} "
        f"(Demo only — not a medical diagnosis.)"
    )


# ── OpenAI tool schemas ───────────────────────────────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_heart_risk",
            "description": (
                "Estimate cardiac risk level from age, systolic blood pressure, "
                "and smoking status. Call this when the user supplies those values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer", "description": "Patient age in years"},
                    "systolic_bp": {
                        "type": "integer",
                        "description": "Systolic blood pressure in mmHg",
                    },
                    "smokes": {
                        "type": "boolean",
                        "description": "True if the patient smokes",
                    },
                },
                "required": ["age", "systolic_bp", "smokes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_ecg_data",
            "description": (
                "Analyse a list of ECG/pulse numeric readings and return a "
                "simple status string. Call this when the user provides a list "
                "of numbers from a watch or ECG device."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_array": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Array of numeric ECG or pulse readings",
                    }
                },
                "required": ["signal_array"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "calculate_heart_risk": lambda args: calculate_heart_risk(
        args["age"], args["systolic_bp"], args["smokes"]
    ),
    "analyze_ecg_data": lambda args: analyze_ecg_data(args["signal_array"]),
}
