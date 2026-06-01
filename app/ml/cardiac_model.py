"""
ML cardiac-risk classifier (scikit-learn).

A LogisticRegression pipeline trained on ~2000 synthetic patients. Features are
``[age, systolic_bp, smokes]`` — the same inputs the sidebar already collects —
so this slots in beside the point-based ``risk_tool.calculate_cardiac_risk``.
Labels come from a noisy logistic ground-truth so the model learns a non-trivial
boundary rather than memorising a rule.

This *complements* the deterministic point-based scorer; both are kept (see
docs/architecture.md on parallel risk calculators).
"""

import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
_ARTIFACT_PATH = os.path.join(_ARTIFACT_DIR, "cardiac_model.joblib")
_RANDOM_STATE = 42

# Module-level cache so we train/load at most once per process.
_model: Pipeline | None = None


def _generate_synthetic_data(n: int = 2000, seed: int = _RANDOM_STATE):
    """Generate synthetic (features, labels) for cardiac risk.

    Returns X of shape (n, 3) with columns [age, systolic_bp, smokes] and a
    binary label derived from a noisy logistic combination of the features.
    """
    rng = np.random.default_rng(seed)

    age = rng.normal(55, 15, n).clip(18, 95)
    systolic_bp = rng.normal(130, 20, n).clip(80, 220)
    smokes = rng.integers(0, 2, n)

    # Latent risk: rises with age, BP, and smoking. Coefficients are arbitrary
    # but produce a sensible, learnable boundary once noise is added.
    z = (
        0.06 * (age - 55)
        + 0.05 * (systolic_bp - 130)
        + 1.2 * smokes
        - 0.5
    )
    prob = 1.0 / (1.0 + np.exp(-z))
    label = (rng.random(n) < prob).astype(int)

    X = np.column_stack([age, systolic_bp, smokes])
    return X, label


def _build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(max_iter=1000, random_state=_RANDOM_STATE),
            ),
        ]
    )


def train(save: bool = True) -> Pipeline:
    """Train the cardiac-risk model on synthetic data and optionally persist it."""
    X, y = _generate_synthetic_data()
    pipeline = _build_pipeline()
    pipeline.fit(X, y)
    if save:
        os.makedirs(_ARTIFACT_DIR, exist_ok=True)
        joblib.dump(pipeline, _ARTIFACT_PATH)
    return pipeline


def _get_model() -> Pipeline:
    """Return the cached model, loading the artifact or training on first use."""
    global _model
    if _model is not None:
        return _model
    if os.path.exists(_ARTIFACT_PATH):
        _model = joblib.load(_ARTIFACT_PATH)
    else:
        _model = train(save=True)
    return _model


def _tier(prob: float) -> tuple[str, str]:
    if prob >= 0.66:
        return "High Risk", "Strongly recommend urgent cardiology consultation."
    if prob >= 0.33:
        return "Moderate Risk", "Consider a full cardiac work-up with your GP."
    return "Low Risk", "Maintain healthy lifestyle; routine check-ups advised."


def predict_cardiac_risk_ml(age: int, systolic_bp: int, smokes: bool) -> str:
    """Return an ML-predicted cardiac-risk summary string.

    Mirrors the tone/structure of ``risk_tool.calculate_cardiac_risk`` so the two
    can be presented side by side.
    """
    age = int(age)
    systolic_bp = int(systolic_bp)
    smokes = int(bool(smokes))

    model = _get_model()
    features = np.array([[age, systolic_bp, smokes]], dtype=float)
    prob = float(model.predict_proba(features)[0, 1])
    tier, advice = _tier(prob)

    smoking_note = "smoker" if smokes else "non-smoker"
    return (
        f"ML Cardiac Risk: {prob * 100:.0f}% probability — {tier}. "
        f"Inputs: age {age}, BP {systolic_bp} mmHg, {smoking_note}. {advice} "
        f"(Demo scikit-learn model trained on synthetic data — not a clinical diagnosis.)"
    )
