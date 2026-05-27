# Cardiac Health PoC — Architecture

## Overview

A Streamlit chat app augmented with deterministic safety guardrails, a vision
analysis pipeline for uploaded ECG images, and a point-based cardiac risk
scorer — all wired to GPT-4o-mini via OpenAI function calling.

```
User browser
    │
    ├─► Sidebar: ECG file upload (jpg/png/pdf)
    │       └─► file_processor.convert_file_to_image()   [PIL / pdf2image → base64]
    │               └─► ecg_vision_tool.analyze_ecg_visual()  [gpt-4o, structured JSON]
    │                       └─► injected into chat as a user turn → LLM explanation
    │
    ├─► Sidebar: Quick Risk Form
    │       └─► risk_tool.calculate_cardiac_risk()  [pure Python, no LLM]
    │
    └─► Chat input
            │
            ├─► emergency_triage.check_emergency_triggers()   ← FIRST, before LLM
            │       True  → st.error(EMERGENCY_MESSAGE), return
            │       False → continue
            │
            └─► OpenAI Chat (gpt-4o-mini)
                    tools=[calculate_cardiac_risk, analyze_ecg_data]
                    │
                    ├─► calculate_cardiac_risk()   [point-based scorer]
                    └─► analyze_ecg_data()         [numeric array heuristic]
```

## Module map

| Path | Purpose |
|------|---------|
| `app.py` | Streamlit entry point, session state, routing |
| `app/tools/emergency_triage.py` | Deterministic keyword guard (Phase 2) |
| `app/tools/risk_tool.py` | Point-based cardiac risk scorer (Phase 4) |
| `app/tools/ecg_vision_tool.py` | gpt-4o vision analysis → structured dict (Phase 3.2) |
| `app/utils/file_processor.py` | JPEG/PNG/PDF → base64 PNG (Phase 3.1) |
| `tools.py` | Legacy numeric-array ECG + original risk calculator (Phase 1) |
| `tests/` | pytest unit tests for all modules |
| `docs/` | This documentation |

## Key design decisions

**Emergency check is purely deterministic.** It runs before every LLM call
so a confused or jailbroken model cannot suppress it.

**Vision tool receives a client argument.** This enables full unit testing
without network access via a mock OpenAI client.

**File processor always outputs PNG.** The vision API receives a consistent
format regardless of the upload type (JPEG, PNG, or PDF first page).

**Tool dispatch in app.py is separate from tools.py.** The new
`calculate_cardiac_risk` from `app/tools/risk_tool.py` uses a different
point-based formula than the original `calculate_heart_risk` in `tools.py`.
Both are kept so neither regresses.
