"""Tests for app/ml/cardiac_model.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re

from app.ml.cardiac_model import predict_cardiac_risk_ml, train


class TestCardiacModel:
    def test_returns_string(self):
        assert isinstance(predict_cardiac_risk_ml(40, 120, False), str)

    def test_contains_tier(self):
        result = predict_cardiac_risk_ml(40, 120, False)
        assert any(t in result for t in ("Low Risk", "Moderate Risk", "High Risk"))

    def test_contains_probability(self):
        result = predict_cardiac_risk_ml(40, 120, False)
        assert "%" in result

    def test_contains_disclaimer(self):
        result = predict_cardiac_risk_ml(40, 120, False)
        assert "demo" in result.lower()

    def test_high_risk_profile_scores_higher_than_low(self):
        def prob(text):
            return float(re.search(r"(\d+)% probability", text).group(1))

        low = predict_cardiac_risk_ml(25, 110, False)
        high = predict_cardiac_risk_ml(75, 180, True)
        assert prob(high) > prob(low)

    def test_string_coercion(self):
        result = predict_cardiac_risk_ml("58", "150", True)
        assert "%" in result

    def test_deterministic_with_fixed_seed(self):
        # Two freshly-trained models on the same seed should agree.
        m1 = train(save=False)
        m2 = train(save=False)
        import numpy as np

        feats = np.array([[60.0, 150.0, 1.0]])
        assert abs(m1.predict_proba(feats)[0, 1] - m2.predict_proba(feats)[0, 1]) < 1e-9
