# Presentation Notes — Cardiac Health Assistant PoC

## Opening (30s)

A heart-health chatbot that does three things well: catches emergencies before
they ever reach the LLM, scores cardiac risk from a couple of numbers, and reads
an ECG image. Built as a Streamlit app, agent loop powered by OpenAI function
calling.

## The problem

- Cardiac symptoms can be life-threatening — a chatbot that hesitates or
  hallucinates is dangerous.
- Users arrive with mixed inputs: free-text symptoms, vitals, wearable data,
  and ECG printouts (often PDFs).
- We need **deterministic safety**, **structured tool outputs**, and a **single
  conversational surface**.

## Architecture at a glance

```
User input ──► Emergency keyword check ──► (red alert, stop)
            │
            └─► LLM (gpt-4o-mini) with bound tools:
                  • calculate_cardiac_risk   (rule-based scoring)
                  • analyze_ecg_data         (numeric array stats)
                  • analyze_ecg_visual       (multimodal vision call)
File upload ─► PDF/PNG → Base64 → analyze_ecg_visual → findings injected
                                                       into chat as a user turn
```

Key decision: the **emergency check runs first, outside the agent loop**. The
LLM is never asked whether something is an emergency — that's a deterministic
keyword match. If it fires, we short-circuit with `st.error()` and a 911/102
prompt.

## Demo script (3 inputs)

1. **"I'm having severe crushing chest pain right now"**
   → emergency intercept fires, big red banner, no LLM call.
2. **"I'm 58, BP 145, and I smoke. What's my risk?"**
   → LLM calls `calculate_cardiac_risk(58, 145, true)` → "High Risk" tier →
   conversational explanation.
3. **Upload an ECG PNG/PDF**
   → `convert_file_to_image` → Base64 → `analyze_ecg_visual` → structured
   JSON `{heart_rate_bpm, rhythm_classification, observations}` → chatbot
   explains the rhythm in plain language.

## What the LLM does *not* decide

- Whether a phrase is an emergency.
- The risk-tier thresholds (pure Python).
- How to parse a PDF (Pillow + pdf2image).

Everything safety- or correctness-critical is deterministic. The LLM is the
orchestrator and translator, not the judge.

## Tradeoffs / limitations

- Keyword matching misses paraphrases ("my chest is being squeezed"). An
  embedding-based classifier would help, but adds latency and a new failure
  mode.
- Risk scoring is intentionally crude — Framingham-style, not a real model.
- Vision analysis trusts whatever the multimodal model returns; we surface the
  raw JSON in `st.info` so a clinician can sanity-check it.
- Single-session state only (Streamlit `session_state`) — no persistence.

## What I'd add next

- Structured logging of every tool call for audit.
- A second deterministic guardrail: if the LLM ever returns the word
  "diagnose" or makes a definitive claim, append the demo disclaimer.
- Replace `gpt-4o-mini` with a routed model: cheap chat default, vision-capable
  model only when an image is in the conversation.
- Eval harness over the three demo inputs in CI so prompt changes can't
  regress safety behaviour.

## Closing line

Safety is deterministic, intelligence is the LLM, and the user only sees one
chat box. That's the whole pitch.
