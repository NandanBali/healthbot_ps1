# Cardiac Health PoC — Architecture

## Overview

A Streamlit chat app augmented with deterministic safety guardrails, a vision
analysis pipeline for uploaded ECG images, a point-based cardiac risk scorer,
trained scikit-learn models (ML cardiac risk + mood classification), and
validated mental-health screeners — all wired to GPT-4o-mini via OpenAI
function calling.

```
User browser
    │
    ├─► Sidebar: ECG file upload (jpg/png/pdf)
    │       └─► file_processor.convert_file_to_image()   [PIL / pdf2image → base64]
    │               └─► ecg_vision_tool.analyze_ecg_visual()  [gpt-4o, structured JSON]
    │                       └─► injected into chat as a user turn → LLM explanation
    │
    ├─► Sidebar: Quick Risk Form
    │       ├─► risk_tool.calculate_cardiac_risk()      [pure Python, no LLM]
    │       └─► cardiac_model.predict_cardiac_risk_ml() [scikit-learn model]
    │
    ├─► Sidebar: PHQ-9 / GAD-7 screeners
    │       └─► mental_health_screeners.score_phq9() / score_gad7()
    │
    ├─► Sidebar: Mood Tracker
    │       └─► mood_model.analyze_mood()  [scikit-learn] → session trend chart
    │
    └─► Chat input
            │
            ├─► emergency_triage.check_emergency_triggers()   ← FIRST, before LLM
            │       True → st.error(EMERGENCY_MESSAGE), return
            │
            ├─► crisis_triage.check_crisis_triggers()         ← also before LLM
            │       True → st.error(CRISIS_MESSAGE), return
            │
            ├─► mood_model.analyze_mood()  → append to mood_history
            │
            └─► OpenAI Chat (gpt-4o-mini)
                    tools=[calculate_cardiac_risk, analyze_ecg_data,
                           predict_cardiac_risk_ml, score_phq9,
                           score_gad7, analyze_mood]
```

## Module map

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit entry point, session state, routing |
| `app/tools/emergency_triage.py` | Deterministic cardiac-emergency keyword guard (Phase 2) |
| `app/tools/crisis_triage.py` | Deterministic self-harm/suicide keyword guard |
| `app/tools/risk_tool.py` | Point-based cardiac risk scorer (Phase 4) |
| `app/tools/mental_health_screeners.py` | PHQ-9 + GAD-7 scoring & interpretation |
| `app/tools/ecg_vision_tool.py` | gpt-4o vision analysis → structured dict (Phase 3.2) |
| `app/ml/cardiac_model.py` | scikit-learn cardiac-risk classifier |
| `app/ml/mood_model.py` | scikit-learn text mood/emotion classifier |
| `app/ml/train.py` | CLI to (re)train models and write artifacts |
| `app/utils/file_processor.py` | JPEG/PNG/PDF → base64 PNG (Phase 3.1) |
| `tools.py` | Legacy numeric-array ECG + original risk calculator (Phase 1) |
| `tests/` | pytest unit tests for all modules |
| `docs/` | This documentation |

## Key design decisions

**Safety checks are purely deterministic.** Both the cardiac-emergency and the
mental-health crisis guards run before every LLM call (plain string matching),
so a confused or jailbroken model cannot suppress them.

**ML models train on synthetic data, no network.** `cardiac_model` and
`mood_model` generate their training data in-process, persist to
`app/ml/artifacts/*.joblib`, and train-on-first-use if an artifact is missing.
A fixed `random_state` keeps results deterministic for tests. The ML cardiac
predictor complements — does not replace — the point-based scorer.

**Vision tool receives a client argument.** This enables full unit testing
without network access via a mock OpenAI client.

**File processor always outputs PNG.** The vision API receives a consistent
format regardless of the upload type (JPEG, PNG, or PDF first page).

**Tool dispatch in app.py is separate from tools.py.** The new
`calculate_cardiac_risk` from `app/tools/risk_tool.py` uses a different
point-based formula than the original `calculate_heart_risk` in `tools.py`.
Both are kept so neither regresses.
