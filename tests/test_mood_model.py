"""Tests for app/ml/mood_model.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.ml.mood_model import analyze_mood, analyze_mood_text


class TestMoodModel:
    def test_returns_dict_with_keys(self):
        result = analyze_mood("i feel calm and relaxed")
        assert "label" in result and "score" in result

    def test_score_in_unit_range(self):
        result = analyze_mood("i feel great today")
        assert 0.0 <= result["score"] <= 1.0

    def test_anxious_text_negative_label(self):
        result = analyze_mood("i feel so anxious and worried about everything")
        assert result["label"] in {"anxious", "stressed", "sad"}

    def test_sad_text_negative_label(self):
        result = analyze_mood("i feel empty hopeless and depressed")
        assert result["label"] in {"anxious", "stressed", "sad"}

    def test_positive_text_non_negative_label(self):
        result = analyze_mood("i am so happy and grateful and joyful")
        assert result["label"] in {"positive", "calm"}

    def test_empty_string(self):
        result = analyze_mood("")
        assert result["score"] == 0.0

    def test_text_wrapper_returns_string(self):
        out = analyze_mood_text("i feel anxious")
        assert isinstance(out, str)
        assert "mood" in out.lower()
