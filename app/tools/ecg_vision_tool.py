"""
Phase 3.2 — Vision-based ECG analysis via a multimodal LLM.

Sends a base64-encoded image to gpt-4o and returns structured JSON
with the keys the task specifies.
"""

import json
import os
import re

from openai import OpenAI

_SYSTEM_PROMPT = (
    "You are an expert diagnostic assistant examining a visual ECG strip or "
    "automated report image. Extract any explicitly printed text metrics "
    "(e.g., 'HR: 82 BPM', 'PR Interval', 'Sinus Rhythm'). If no text "
    "parameters are printed, visually estimate if the QRS complexes appear "
    "regularly spaced or abnormally irregular. "
    "Return a strict JSON response with keys: "
    "'estimated_bpm' (integer or null), "
    "'rhythm_status' (one of: 'Normal', 'Irregular', 'Unclear'), "
    "and 'visual_summary' (brief textual breakdown). "
    "Output ONLY valid JSON — no markdown fences, no extra text."
)

_VALID_RHYTHM_STATUSES = {"Normal", "Irregular", "Unclear"}


def analyze_ecg_visual(base64_image: str, client: OpenAI | None = None) -> dict:
    """
    Send a base64-encoded ECG image to gpt-4o and return structured findings.

    Returns a dict with keys:
      heart_rate_bpm         int | None
      rhythm_classification  str
      observations           str

    Raises RuntimeError if the API response cannot be parsed.
    """
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Analyse this ECG image and return the JSON as instructed.",
                    },
                ],
            },
        ],
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if the model wraps with them anyway
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Vision model returned non-JSON: {raw!r}") from exc

    bpm = parsed.get("estimated_bpm")
    rhythm = parsed.get("rhythm_status", "Unclear")
    if rhythm not in _VALID_RHYTHM_STATUSES:
        rhythm = "Unclear"

    return {
        "heart_rate_bpm": int(bpm) if bpm is not None else None,
        "rhythm_classification": _expand_rhythm(rhythm),
        "observations": parsed.get("visual_summary", "No summary provided."),
    }


def _expand_rhythm(status: str) -> str:
    mapping = {
        "Normal": "Normal Sinus Rhythm",
        "Irregular": "Irregular Rhythm — further evaluation recommended",
        "Unclear": "Rhythm Unclear — image quality may be insufficient",
    }
    return mapping.get(status, status)
