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
