# Running the App

## Prerequisites

- Python 3.10+
- OpenAI API key (gpt-4o-mini for chat; gpt-4o for ECG vision)
- `poppler` system package (only needed for PDF uploads)

```bash
# Fedora / RHEL
sudo dnf install poppler-utils

# Ubuntu / Debian
sudo apt install poppler-utils
```

## Install Python dependencies

```bash
pip install openai streamlit Pillow pdf2image
```

## Set your API key

```bash
export OPENAI_API_KEY="sk-..."
```

## Start the app

```bash
streamlit run app.py
```

Open http://localhost:8501.

---

## Integration test matrix (Phase 5.3)

| Scenario | Input | Expected |
|----------|-------|----------|
| Emergency lockout | `I'm having severe crushing chest pain right now` | Red `st.error` banner, LLM not called |
| Risk calculator | Form: age=58, BP=145, smoker | `High Risk` (score 7/7) |
| ECG file upload | Upload a PNG/PDF ECG → click Analyse ECG | Image converted to base64, gpt-4o called, chat shows rhythm summary |

---

## Run with Docker

```bash
# Build
docker build -t healthbot .

# Run (pass API key from host env)
docker run --rm -p 8501:8501 -e OPENAI_API_KEY="$OPENAI_API_KEY" healthbot

# Or via docker-compose
export OPENAI_API_KEY="sk-..."
docker compose up --build
```

The image is based on `python:3.11-slim` and includes `poppler-utils` for PDF uploads. Open http://localhost:8501.

To run the tests inside the container:

```bash
docker run --rm healthbot pytest tests/ -v
```

---

## Run tests

```bash
pytest tests/ -v
```

All 105 tests are pure-Python and require no API key.

| Test file | Coverage |
|-----------|----------|
| `tests/test_emergency_triage.py` | Emergency keyword matching |
| `tests/test_risk_tool.py` | Point-based risk scorer including all tier boundaries |
| `tests/test_file_processor.py` | PNG/JPEG/PDF conversion, error cases |
| `tests/test_ecg_vision_tool.py` | Vision tool with mocked OpenAI client |
| `tests/test_tools.py` | Legacy Phase 1 tools |
