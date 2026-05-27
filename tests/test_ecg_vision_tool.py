"""
Tests for app/tools/ecg_vision_tool.py

The vision tool makes an OpenAI API call, so all tests mock the client.
We test:
  - correct JSON parsing and field mapping
  - rhythm expansion
  - graceful handling of bad JSON from the model
  - markdown fence stripping
  - null BPM handling
  - invalid rhythm_status falls back to Unclear
"""

import sys
import os
import json
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from app.tools.ecg_vision_tool import analyze_ecg_visual, _expand_rhythm


def _make_client(response_text: str) -> MagicMock:
    """Return a mock OpenAI client that returns response_text as the model's message."""
    mock_message = MagicMock()
    mock_message.content = response_text

    mock_choice = MagicMock()
    mock_choice.message = mock_message

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


class TestExpandRhythm:
    def test_normal(self):
        assert "Normal Sinus Rhythm" in _expand_rhythm("Normal")

    def test_irregular(self):
        assert "Irregular" in _expand_rhythm("Irregular")

    def test_unclear(self):
        assert "Unclear" in _expand_rhythm("Unclear")

    def test_unknown_passthrough(self):
        assert _expand_rhythm("Unknown") == "Unknown"


class TestAnalyzeEcgVisual:
    # ── Happy path ────────────────────────────────────────────────────────────

    def test_normal_rhythm_parsed(self):
        payload = json.dumps({
            "estimated_bpm": 78,
            "rhythm_status": "Normal",
            "visual_summary": "Regular QRS complexes visible.",
        })
        client = _make_client(payload)
        result = analyze_ecg_visual("dummyb64==", client=client)
        assert result["heart_rate_bpm"] == 78
        assert "Normal Sinus Rhythm" in result["rhythm_classification"]
        assert "Regular QRS" in result["observations"]

    def test_irregular_rhythm(self):
        payload = json.dumps({
            "estimated_bpm": 95,
            "rhythm_status": "Irregular",
            "visual_summary": "Uneven RR intervals detected.",
        })
        client = _make_client(payload)
        result = analyze_ecg_visual("dummyb64==", client=client)
        assert "Irregular" in result["rhythm_classification"]
        assert result["heart_rate_bpm"] == 95

    def test_unclear_rhythm(self):
        payload = json.dumps({
            "estimated_bpm": None,
            "rhythm_status": "Unclear",
            "visual_summary": "Image too blurry to assess.",
        })
        client = _make_client(payload)
        result = analyze_ecg_visual("dummyb64==", client=client)
        assert result["heart_rate_bpm"] is None
        assert "Unclear" in result["rhythm_classification"]

    def test_null_bpm_stays_none(self):
        payload = json.dumps({
            "estimated_bpm": None,
            "rhythm_status": "Normal",
            "visual_summary": "No BPM printed.",
        })
        client = _make_client(payload)
        result = analyze_ecg_visual("dummyb64==", client=client)
        assert result["heart_rate_bpm"] is None

    def test_integer_bpm_coercion(self):
        payload = json.dumps({
            "estimated_bpm": "82",  # model returns string
            "rhythm_status": "Normal",
            "visual_summary": "Fine.",
        })
        client = _make_client(payload)
        result = analyze_ecg_visual("dummyb64==", client=client)
        assert isinstance(result["heart_rate_bpm"], int)
        assert result["heart_rate_bpm"] == 82

    # ── Markdown fence stripping ──────────────────────────────────────────────

    def test_strips_markdown_fences(self):
        inner = json.dumps({
            "estimated_bpm": 70,
            "rhythm_status": "Normal",
            "visual_summary": "Fine.",
        })
        fenced = f"```json\n{inner}\n```"
        client = _make_client(fenced)
        result = analyze_ecg_visual("dummyb64==", client=client)
        assert result["heart_rate_bpm"] == 70

    def test_strips_plain_fences(self):
        inner = json.dumps({
            "estimated_bpm": 65,
            "rhythm_status": "Normal",
            "visual_summary": "Ok.",
        })
        fenced = f"```\n{inner}\n```"
        client = _make_client(fenced)
        result = analyze_ecg_visual("dummyb64==", client=client)
        assert result["heart_rate_bpm"] == 65

    # ── Invalid rhythm falls back to Unclear ──────────────────────────────────

    def test_unknown_rhythm_status_becomes_unclear(self):
        payload = json.dumps({
            "estimated_bpm": 80,
            "rhythm_status": "Atrial Fibrillation",  # not in allowed set
            "visual_summary": "AF pattern.",
        })
        client = _make_client(payload)
        result = analyze_ecg_visual("dummyb64==", client=client)
        assert "Unclear" in result["rhythm_classification"]

    # ── Error handling ────────────────────────────────────────────────────────

    def test_non_json_raises_runtime_error(self):
        client = _make_client("Sorry, I cannot analyse this image.")
        with pytest.raises(RuntimeError, match="non-JSON"):
            analyze_ecg_visual("dummyb64==", client=client)

    def test_empty_response_raises(self):
        client = _make_client("")
        with pytest.raises((RuntimeError, json.JSONDecodeError)):
            analyze_ecg_visual("dummyb64==", client=client)

    # ── Return structure ──────────────────────────────────────────────────────

    def test_result_has_required_keys(self):
        payload = json.dumps({
            "estimated_bpm": 72,
            "rhythm_status": "Normal",
            "visual_summary": "All good.",
        })
        client = _make_client(payload)
        result = analyze_ecg_visual("dummyb64==", client=client)
        assert set(result.keys()) == {"heart_rate_bpm", "rhythm_classification", "observations"}

    def test_result_is_dict(self):
        payload = json.dumps({
            "estimated_bpm": 72,
            "rhythm_status": "Normal",
            "visual_summary": "All good.",
        })
        client = _make_client(payload)
        result = analyze_ecg_visual("dummyb64==", client=client)
        assert isinstance(result, dict)

    # ── API is called with the image ──────────────────────────────────────────

    def test_api_called_with_base64_image(self):
        payload = json.dumps({
            "estimated_bpm": 72,
            "rhythm_status": "Normal",
            "visual_summary": "Ok.",
        })
        client = _make_client(payload)
        analyze_ecg_visual("MYBASE64==", client=client)
        call_kwargs = client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        # Find user message content
        user_msg = next(m for m in messages if m["role"] == "user")
        content = user_msg["content"]
        image_parts = [c for c in content if c.get("type") == "image_url"]
        assert len(image_parts) == 1
        assert "MYBASE64==" in image_parts[0]["image_url"]["url"]

    def test_model_is_gpt4o(self):
        payload = json.dumps({
            "estimated_bpm": 72,
            "rhythm_status": "Normal",
            "visual_summary": "Ok.",
        })
        client = _make_client(payload)
        analyze_ecg_visual("b64==", client=client)
        call_kwargs = client.chat.completions.create.call_args
        model = call_kwargs.kwargs.get("model") or call_kwargs.args[1]
        assert model == "gpt-4o"

    # ── Missing OPENAI_API_KEY when no client passed ──────────────────────────

    def test_no_key_no_client_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            # Remove key if present
            env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
                    analyze_ecg_visual("b64==")
