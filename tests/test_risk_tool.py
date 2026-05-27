"""Tests for app/tools/risk_tool.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.tools.risk_tool import calculate_cardiac_risk


class TestCalculateCardiacRisk:
    # ── Score = 0: Low Risk ───────────────────────────────────────────────────

    def test_young_healthy_nonsmoker(self):
        result = calculate_cardiac_risk(30, 120, False)
        assert "Low Risk" in result

    def test_age_exactly_55_not_triggered(self):
        # threshold is > 55, so 55 alone doesn't add points
        result = calculate_cardiac_risk(55, 120, False)
        assert "Low Risk" in result

    def test_bp_exactly_140_not_triggered(self):
        # threshold is > 140
        result = calculate_cardiac_risk(40, 140, False)
        assert "Low Risk" in result

    # ── Score = 2: Low Risk (only age points) ─────────────────────────────────

    def test_age_56_no_other_factors(self):
        result = calculate_cardiac_risk(56, 120, False)
        assert "Low Risk" in result

    # ── Score = 3: Moderate Risk (BP only) ───────────────────────────────────

    def test_high_bp_only(self):
        result = calculate_cardiac_risk(40, 145, False)
        assert "Moderate Risk" in result

    # ── Score = 4: Moderate Risk (age + bp OR age + smoking, etc) ─────────────

    def test_age_and_smoking(self):
        # 56 → +2, smokes → +2 = 4 → Moderate
        result = calculate_cardiac_risk(56, 130, True)
        assert "Moderate Risk" in result

    def test_age_and_high_bp(self):
        # 56 → +2, bp 145 → +3 = 5 → High
        result = calculate_cardiac_risk(56, 145, False)
        assert "High Risk" in result

    # ── Score = 5+: High Risk ─────────────────────────────────────────────────

    def test_all_three_factors(self):
        # age 58 → +2, BP 150 → +3, smokes → +2 = 7
        result = calculate_cardiac_risk(58, 150, True)
        assert "High Risk" in result

    def test_task_example_58_145_smoker(self):
        # From TASK2 Phase 5.3: age 58, BP 145, smoker → High Risk
        result = calculate_cardiac_risk(58, 145, True)
        assert "High Risk" in result

    def test_bp_and_smoking(self):
        # bp 141 → +3, smokes → +2 = 5 → High
        result = calculate_cardiac_risk(40, 141, True)
        assert "High Risk" in result

    # ── Return type and structure ─────────────────────────────────────────────

    def test_returns_string(self):
        assert isinstance(calculate_cardiac_risk(40, 120, False), str)

    def test_contains_score(self):
        result = calculate_cardiac_risk(40, 120, False)
        assert "/7" in result

    def test_contains_disclaimer(self):
        result = calculate_cardiac_risk(40, 120, False)
        assert "demo" in result.lower() or "not a clinical" in result.lower()

    def test_string_coercion(self):
        result = calculate_cardiac_risk("58", "150", True)
        assert "High Risk" in result

    # ── Factor listing ────────────────────────────────────────────────────────

    def test_no_factors_listed_for_young_healthy(self):
        result = calculate_cardiac_risk(30, 120, False)
        assert "no elevated risk factors" in result

    def test_smoker_factor_listed(self):
        result = calculate_cardiac_risk(30, 120, True)
        assert "smoker" in result

    def test_age_factor_listed_when_over_55(self):
        result = calculate_cardiac_risk(60, 120, False)
        assert "age 60" in result

    def test_bp_factor_listed_when_over_140(self):
        result = calculate_cardiac_risk(40, 145, False)
        assert "145" in result

    # ── Boundary: exactly at threshold+1 ─────────────────────────────────────

    def test_age_56_triggers_points(self):
        r_no_age = calculate_cardiac_risk(55, 120, False)
        r_age = calculate_cardiac_risk(56, 120, False)
        # 56 adds 2 pts; with nothing else it's still Low Risk but score differs
        assert "0/7" in r_no_age
        assert "2/7" in r_age

    def test_bp_141_triggers_points(self):
        r_no = calculate_cardiac_risk(40, 140, False)
        r_yes = calculate_cardiac_risk(40, 141, False)
        assert "0/7" in r_no
        assert "3/7" in r_yes
