"""Tests for the cardiac PoC tool functions."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from tools import (
    analyze_ecg_data,
    calculate_heart_risk,
    check_emergency,
    TOOL_SCHEMAS,
    TOOL_DISPATCH,
)


# ── check_emergency ───────────────────────────────────────────────────────────


class TestCheckEmergency:
    def test_chest_pain_triggers(self):
        assert check_emergency("I have bad chest pain") is True

    def test_heart_attack_triggers(self):
        assert check_emergency("I think I'm having a heart attack") is True

    def test_shortness_of_breath_triggers(self):
        assert check_emergency("I have shortness of breath") is True

    def test_case_insensitive(self):
        assert check_emergency("CHEST PAIN right now") is True

    def test_normal_query_does_not_trigger(self):
        assert check_emergency("I am 60, my blood pressure is 150") is False

    def test_ecg_list_does_not_trigger(self):
        assert check_emergency("Can you look at this data: [1, 2, 9, 1]") is False

    def test_empty_string(self):
        assert check_emergency("") is False

    def test_cardiac_arrest_triggers(self):
        assert check_emergency("He had a cardiac arrest") is True


# ── calculate_heart_risk ──────────────────────────────────────────────────────


class TestCalculateHeartRisk:
    def test_high_risk_old_high_bp(self):
        result = calculate_heart_risk(60, 150, False)
        assert "High" in result

    def test_high_risk_smoker(self):
        result = calculate_heart_risk(46, 131, True)
        assert "High" in result

    def test_medium_risk_over_40(self):
        result = calculate_heart_risk(45, 120, False)
        assert "Medium" in result

    def test_medium_risk_elevated_bp(self):
        result = calculate_heart_risk(35, 135, False)
        assert "Medium" in result

    def test_low_risk_young_healthy(self):
        result = calculate_heart_risk(30, 115, False)
        assert "Low" in result

    def test_result_is_string(self):
        assert isinstance(calculate_heart_risk(40, 120, False), str)

    def test_contains_demo_disclaimer(self):
        result = calculate_heart_risk(50, 140, False)
        assert "demo" in result.lower() or "not medical" in result.lower()

    def test_string_inputs_accepted(self):
        # Tool dispatch passes JSON-parsed values which may be int already,
        # but the function should also handle string coercion.
        result = calculate_heart_risk("60", "150", False)
        assert "High" in result


# ── analyze_ecg_data ──────────────────────────────────────────────────────────


class TestAnalyzeEcgData:
    def test_normal_signal(self):
        # Average 7.2, scaled BPM = 72 → normal
        signal = [7, 7, 7, 7, 7, 7, 7, 7, 7, 7]
        result = analyze_ecg_data(signal)
        assert "normal" in result.lower()

    def test_watch_data_from_task(self):
        result = analyze_ecg_data([1, 2, 9, 1, 2, 9])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_list(self):
        result = analyze_ecg_data([])
        assert "No ECG data" in result

    def test_result_is_string(self):
        assert isinstance(analyze_ecg_data([5, 6, 7]), str)

    def test_contains_demo_disclaimer(self):
        result = analyze_ecg_data([5, 6, 7])
        assert "demo" in result.lower() or "not a medical" in result.lower()

    def test_high_values_detected(self):
        # Very high average → elevated BPM
        result = analyze_ecg_data([15, 16, 17, 18])
        assert "elevated" in result.lower() or "tachycardia" in result.lower()

    def test_low_values_detected(self):
        # Very low average → low BPM
        result = analyze_ecg_data([1, 1, 1, 1])
        assert "low" in result.lower() or "bradycardia" in result.lower()


# ── tool schemas & dispatch ───────────────────────────────────────────────────


class TestToolSchemas:
    def test_two_schemas_defined(self):
        assert len(TOOL_SCHEMAS) == 2

    def test_schema_names(self):
        names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert names == {"calculate_heart_risk", "analyze_ecg_data"}

    def test_dispatch_keys_match_schemas(self):
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert set(TOOL_DISPATCH.keys()) == schema_names

    def test_dispatch_calculate_risk(self):
        result = TOOL_DISPATCH["calculate_heart_risk"](
            {"age": 60, "systolic_bp": 150, "smokes": False}
        )
        assert "High" in result

    def test_dispatch_analyze_ecg(self):
        result = TOOL_DISPATCH["analyze_ecg_data"]({"signal_array": [7, 7, 7]})
        assert isinstance(result, str)
