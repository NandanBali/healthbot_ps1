"""Tests for app/tools/crisis_triage.py"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.tools.crisis_triage import (
    CRISIS_MESSAGE,
    CRISIS_TRIGGERS,
    check_crisis_triggers,
)


class TestCheckCrisisTriggers:
    # ── Positive matches ──────────────────────────────────────────────────────

    def test_kill_myself(self):
        assert check_crisis_triggers("I want to kill myself") is True

    def test_want_to_die(self):
        assert check_crisis_triggers("sometimes i just want to die") is True

    def test_end_my_life(self):
        assert check_crisis_triggers("thinking about how to end my life") is True

    def test_suicidal(self):
        assert check_crisis_triggers("I have been feeling suicidal lately") is True

    def test_self_harm_hyphen(self):
        assert check_crisis_triggers("urges to self-harm") is True

    def test_self_harm_space(self):
        assert check_crisis_triggers("i engage in self harm") is True

    def test_better_off_dead(self):
        assert check_crisis_triggers("everyone would be better off dead") is True

    # ── Case insensitivity ────────────────────────────────────────────────────

    def test_uppercase(self):
        assert check_crisis_triggers("I WANT TO DIE") is True

    def test_mixed_case(self):
        assert check_crisis_triggers("Kill Myself tonight") is True

    # ── Negative cases ────────────────────────────────────────────────────────

    def test_normal_query(self):
        assert check_crisis_triggers("How do I lower my blood pressure?") is False

    def test_empty_string(self):
        assert check_crisis_triggers("") is False

    def test_whitespace_only(self):
        assert check_crisis_triggers("   ") is False

    def test_unrelated_sad_but_not_crisis(self):
        assert check_crisis_triggers("I had a rough day at work") is False

    # ── Constants sanity ──────────────────────────────────────────────────────

    def test_triggers_list_nonempty(self):
        assert len(CRISIS_TRIGGERS) > 0

    def test_message_contains_988(self):
        assert "988" in CRISIS_MESSAGE
