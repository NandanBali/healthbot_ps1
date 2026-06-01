# Cardiac Health Assistant (PoC)

A Streamlit chatbot demo for cardiac-health triage. Combines a deterministic
emergency intercept, a point-based risk calculator, and a multimodal ECG image
analyser behind an OpenAI function-calling agent loop.

> **Demo only — not a medical device. Always consult a real doctor.**

## Features

- **Emergency intercept** — pre-LLM keyword check that bypasses the model and
  surfaces a 911/102 prompt on phrases like "chest pain" or "can't breathe".
- **Cardiac risk calculator** — point-based scoring over age, systolic BP, and
  smoking status; returns Low / Moderate / High tier.
- **ECG vision tool** — upload an ECG image (JPG/PNG) or PDF; the first page is
  converted to a Base64 image and analysed by a multimodal model that returns
  structured `{heart_rate_bpm, rhythm_classification, observations}`.
- **Legacy numeric ECG tool** — accepts an array of pulse samples and returns a
  simple BPM / variability summary.
- **Agent loop** — both tools are bound to `gpt-4o-mini` via OpenAI function
  calling; the LLM decides when to invoke them.

## Project layout

```
app.py                       Streamlit entry point + agent loop
tools.py                     Legacy numeric ECG tool + emergency keywords
app/tools/emergency_triage.py  Deterministic emergency intercept
app/tools/risk_tool.py         Point-based cardiac risk scoring
app/tools/ecg_vision_tool.py   Multimodal ECG image analysis
app/utils/file_processor.py    Image/PDF → Base64 conversion
tests/                       Pytest suite for each tool
docs/                        Architecture, API reference, running notes
```

## Running locally

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py
```

PDF uploads require `poppler-utils` on the host (already installed in the
Docker image).

## Running with Docker

```bash
export OPENAI_API_KEY=sk-...
docker compose up --build
```

Open http://localhost:8501.

## Tests

```bash
pytest
```

## Tool contract

| Tool | Input | Output |
|---|---|---|
| `calculate_cardiac_risk` | `age:int, systolic_bp:int, smokes:bool` | Risk-tier summary string |
| `analyze_ecg_data` | `signal_array: number[]` | BPM + variability string |
| `analyze_ecg_visual` | Base64 PNG/JPEG | `{heart_rate_bpm, rhythm_classification, observations}` |
| `check_emergency_triggers` | `user_text:str` | `bool` — runs **before** the LLM |
