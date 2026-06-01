"""Tests for app/tools/mental_health_screeners.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.tools.mental_health_screeners import (
    GAD7_QUESTIONS,
    PHQ9_QUESTIONS,
    score_gad7,
    score_phq9,
)


def _phq9(total_for_first_item, item9=0):
    """Build a 9-item answer list summing roughly to a target via item 0."""
    answers = [0] * 9
    answers[8] = item9
    return answers


class TestScorePHQ9:
    def test_question_count(self):
        assert len(PHQ9_QUESTIONS) == 9

    def test_minimal(self):
        assert "Minimal depression" in score_phq9([0] * 9)

    def test_mild(self):
        # sum = 5
        assert "Mild depression" in score_phq9([1, 1, 1, 1, 1, 0, 0, 0, 0])

    def test_moderate(self):
        # sum = 10
        assert "Moderate depression" in score_phq9([2, 2, 2, 2, 2, 0, 0, 0, 0])

    def test_moderately_severe(self):
        # sum = 15
        assert "Moderately severe depression" in score_phq9([3, 3, 3, 3, 3, 0, 0, 0, 0])

    def test_severe(self):
        # sum = 27
        assert "Severe depression" in score_phq9([3] * 9)

    def test_item9_safety_note(self):
        result = score_phq9([0, 0, 0, 0, 0, 0, 0, 0, 2])
        assert "988" in result

    def test_no_safety_note_when_item9_zero(self):
        result = score_phq9([1, 1, 1, 0, 0, 0, 0, 0, 0])
        assert "988" not in result

    def test_disclaimer(self):
        assert "not a diagnosis" in score_phq9([0] * 9).lower()

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            score_phq9([0] * 8)

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            score_phq9([4] + [0] * 8)

    def test_score_string(self):
        assert "/27" in score_phq9([0] * 9)


class TestScoreGAD7:
    def test_question_count(self):
        assert len(GAD7_QUESTIONS) == 7

    def test_minimal(self):
        assert "Minimal anxiety" in score_gad7([0] * 7)

    def test_mild(self):
        # sum = 5
        assert "Mild anxiety" in score_gad7([1, 1, 1, 1, 1, 0, 0])

    def test_moderate(self):
        # sum = 10
        assert "Moderate anxiety" in score_gad7([2, 2, 2, 2, 2, 0, 0])

    def test_severe(self):
        # sum = 21
        assert "Severe anxiety" in score_gad7([3] * 7)

    def test_disclaimer(self):
        assert "not a diagnosis" in score_gad7([0] * 7).lower()

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            score_gad7([0] * 6)

    def test_score_string(self):
        assert "/21" in score_gad7([0] * 7)
