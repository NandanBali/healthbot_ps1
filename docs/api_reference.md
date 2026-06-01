# API Reference

---

## app/tools/emergency_triage.py

### `check_emergency_triggers(user_text: str) -> bool`

Case-insensitive scan for acute cardiac emergency phrases.

**Returns** `True` if any keyword matches, `False` otherwise.

**Triggers include:** chest pain, crushing pressure, left arm pain,
heart attack, can't breathe, cannot breathe, shortness of breath,
cardiac arrest, jaw pain, crushing chest.

### `EMERGENCY_MESSAGE: str`

Pre-formatted display string: _"🚨 EMERGENCY LOGGED: Please call emergency services (911/102) immediately."_

---

## app/tools/risk_tool.py

### `calculate_cardiac_risk(age: int, systolic_bp: int, smokes: bool) -> str`

Point-based cardiac risk scorer.

| Condition | Points |
|-----------|--------|
| age > 55 | +2 |
| systolic_bp > 140 | +3 |
| smokes == True | +2 |

| Total | Tier |
|-------|------|
| 0–2 | Low Risk |
| 3–4 | Moderate Risk |
| 5+ | High Risk |

**Returns** concise summary string with score, tier, factors, advice, and disclaimer.

---

## app/tools/crisis_triage.py

### `check_crisis_triggers(user_text: str) -> bool`

Case-insensitive scan for self-harm / suicidal phrases. Like the emergency
guard, this runs **before** any LLM call and bypasses the model entirely.

**Returns** `True` if any trigger matches, `False` otherwise.

**Triggers include:** kill myself, want to die, end my life, suicidal, self
harm, hurt myself, better off dead, take my own life, no reason to live.

### `CRISIS_MESSAGE: str`

Crisis-hotline display string (US 988; India iCall/AASRA; local emergency
numbers).

---

## app/tools/mental_health_screeners.py

### `score_phq9(answers: list[int]) -> str`

Scores a 9-item PHQ-9 depression screener. Each answer is `0–3`.

| Total | Severity |
|-------|----------|
| 0–4 | Minimal depression |
| 5–9 | Mild depression |
| 10–14 | Moderate depression |
| 15–19 | Moderately severe depression |
| 20–27 | Severe depression |

If item 9 (self-harm thoughts) is endorsed (≥1), the result appends a safety
note with hotline numbers. **Raises** `ValueError` if not exactly 9 answers or
any answer is outside `0–3`.

### `score_gad7(answers: list[int]) -> str`

Scores a 7-item GAD-7 anxiety screener. Each answer is `0–3`.

| Total | Severity |
|-------|----------|
| 0–4 | Minimal anxiety |
| 5–9 | Mild anxiety |
| 10–14 | Moderate anxiety |
| 15–21 | Severe anxiety |

**Raises** `ValueError` if not exactly 7 answers or any answer is outside `0–3`.

### `PHQ9_QUESTIONS`, `GAD7_QUESTIONS`, `LIKERT_LABELS`

Question text lists (9 and 7 items) and the four `0–3` Likert labels, used to
render the sidebar forms.

---

## app/ml/cardiac_model.py

### `predict_cardiac_risk_ml(age: int, systolic_bp: int, smokes: bool) -> str`

ML cardiac-risk prediction. A scikit-learn `StandardScaler` + `LogisticRegression`
pipeline trained on ~2000 synthetic patients (features `[age, systolic_bp, smokes]`).
Returns a string with the predicted probability, a Low/Moderate/High tier
(≥0.66 High, ≥0.33 Moderate, else Low), and a disclaimer.

Complements — does not replace — the point-based `calculate_cardiac_risk`.

### `train(save: bool = True) -> Pipeline`

Trains the model on synthetic data and (by default) persists it to
`app/ml/artifacts/cardiac_model.joblib`. Loaded lazily; trains on first use if
the artifact is absent. Deterministic via fixed `random_state`.

---

## app/ml/mood_model.py

### `analyze_mood(text: str) -> dict`

TF-IDF + `LogisticRegression` text classifier. Returns
`{"label": str, "score": float}` where `label` ∈
`positive / calm / anxious / stressed / sad` and `score` is the predicted-class
confidence in `[0, 1]`. Empty text returns `{"label": "calm", "score": 0.0}`.

### `analyze_mood_text(text: str) -> str`

String-returning wrapper used by the chat function-calling tool.

### `train(save: bool = True) -> Pipeline`

Trains on the in-module labeled phrase dataset; persists to
`app/ml/artifacts/mood_model.joblib`.

---

## app/ml/train.py

CLI entry point: `python -m app.ml.train` (re)trains both models and writes
their artifacts.

---

## app/utils/file_processor.py

### `convert_file_to_image(uploaded_file) -> str`

Converts an uploaded file to a base64-encoded PNG string.

`uploaded_file` must have `.name` (str) and `.read()` (returns bytes).

| Extension | Behaviour |
|-----------|-----------|
| jpg / jpeg | Opens with Pillow |
| png | Opens with Pillow |
| pdf | Converts first page with pdf2image |
| other | Raises `ValueError` |

**Returns** base64 string (no data-URI prefix). Output is always RGB PNG.

**Raises** `ValueError` for unsupported file types or empty PDFs.

---

## app/tools/ecg_vision_tool.py

### `analyze_ecg_visual(base64_image: str, client: OpenAI | None = None) -> dict`

Sends a base64 PNG to gpt-4o with a structured system prompt and returns findings.

**Parameters:**
- `base64_image` — output of `convert_file_to_image`
- `client` — optional `OpenAI` instance (for testing/injection); if `None`, creates one from `OPENAI_API_KEY`

**Returns:**
```python
{
    "heart_rate_bpm": int | None,
    "rhythm_classification": str,   # expanded label
    "observations": str,
}
```

**rhythm_classification values:**
- `"Normal Sinus Rhythm"`
- `"Irregular Rhythm — further evaluation recommended"`
- `"Rhythm Unclear — image quality may be insufficient"`

**Raises:**
- `EnvironmentError` if `OPENAI_API_KEY` unset and no client provided
- `RuntimeError` if the model returns non-JSON

### `_expand_rhythm(status: str) -> str`

Maps `"Normal" | "Irregular" | "Unclear"` → human-friendly label.

---

## tools.py (legacy — Phase 1)

### `check_emergency(user_text: str) -> bool`
### `calculate_heart_risk(age, systolic_bp, smokes) -> str`
### `analyze_ecg_data(signal_array: list) -> str`
### `TOOL_SCHEMAS`, `TOOL_DISPATCH`

See `docs/api_reference.md` Phase 1 section for details.
