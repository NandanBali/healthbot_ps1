"""Tests for app/tools/emergency_triage.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.tools.emergency_triage import check_emergency_triggers, EMERGENCY_TRIGGERS, EMERGENCY_MESSAGE


class TestCheckEmergencyTriggers:
    # ── Positive matches ──────────────────────────────────────────────────────

    def test_chest_pain(self):
        assert check_emergency_triggers("I have chest pain") is True

    def test_crushing_pressure(self):
        assert check_emergency_triggers("There is crushing pressure in my chest") is True

    def test_left_arm_pain(self):
        assert check_emergency_triggers("I feel left arm pain radiating down") is True

    def test_heart_attack(self):
        assert check_emergency_triggers("I think I'm having a heart attack") is True

    def test_cant_breathe_apostrophe(self):
        assert check_emergency_triggers("I can't breathe properly") is True

    def test_cannot_breathe(self):
        assert check_emergency_triggers("I cannot breathe at all") is True

    def test_shortness_of_breath(self):
        assert check_emergency_triggers("severe shortness of breath since morning") is True

    def test_cardiac_arrest(self):
        assert check_emergency_triggers("he went into cardiac arrest") is True

    def test_jaw_pain(self):
        assert check_emergency_triggers("sudden jaw pain and sweating") is True

    def test_crushing_chest(self):
        assert check_emergency_triggers("crushing chest sensation") is True

    # ── Case insensitivity ────────────────────────────────────────────────────

    def test_uppercase_heart_attack(self):
        assert check_emergency_triggers("HEART ATTACK") is True

    def test_mixed_case(self):
        assert check_emergency_triggers("Chest Pain right now") is True

    def test_all_caps_cannot_breathe(self):
        assert check_emergency_triggers("CANNOT BREATHE") is True

    # ── Negative cases ────────────────────────────────────────────────────────

    def test_normal_query(self):
        assert check_emergency_triggers("What is my blood pressure reading?") is False

    def test_risk_query(self):
        assert check_emergency_triggers("I am 60 years old, BP 150, I smoke") is False

    def test_ecg_numbers(self):
        assert check_emergency_triggers("data: [1, 2, 9, 1, 2, 9]") is False

    def test_empty_string(self):
        assert check_emergency_triggers("") is False

    def test_whitespace_only(self):
        assert check_emergency_triggers("   ") is False

    def test_unrelated_medical(self):
        assert check_emergency_triggers("I have a headache today") is False

    # ── Substring boundary checks ─────────────────────────────────────────────

    def test_partial_word_does_not_match(self):
        # "art" is inside "heart attack" but alone it should not match
        assert check_emergency_triggers("I study art") is False

    def test_trigger_in_long_sentence(self):
        assert (
            check_emergency_triggers(
                "My doctor said I need to monitor because chest pain was mentioned last week"
            )
            is True
        )

    # ── Constants sanity checks ───────────────────────────────────────────────

    def test_triggers_list_nonempty(self):
        assert len(EMERGENCY_TRIGGERS) > 0

    def test_emergency_message_contains_911(self):
        assert "911" in EMERGENCY_MESSAGE

    def test_emergency_message_contains_102(self):
        assert "102" in EMERGENCY_MESSAGE
