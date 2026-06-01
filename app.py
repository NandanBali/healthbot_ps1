"""
Cardiac Health PoC — Streamlit chat app.

Phases covered:
  2 — Emergency keyword intercept (deterministic, pre-LLM)
  3 — ECG file upload → image conversion → vision analysis
  4 — Cardiac risk calculator (point-based)
  5 — Master agent with OpenAI function calling

Run with:
    streamlit run app.py

Requires OPENAI_API_KEY set in the environment.
"""

import json
import os

import streamlit as st
from openai import OpenAI

from app.tools.emergency_triage import EMERGENCY_MESSAGE, check_emergency_triggers
from app.tools.crisis_triage import CRISIS_MESSAGE, check_crisis_triggers
from app.tools.ecg_vision_tool import analyze_ecg_visual
from app.tools.risk_tool import calculate_cardiac_risk
from app.tools.mental_health_screeners import (
    GAD7_QUESTIONS,
    LIKERT_LABELS,
    PHQ9_QUESTIONS,
    score_gad7,
    score_phq9,
)
from app.ml.cardiac_model import predict_cardiac_risk_ml
from app.ml.mood_model import analyze_mood, analyze_mood_text
from app.utils.file_processor import convert_file_to_image

# ── Legacy numeric-array ECG tool (Phase 1) ───────────────────────────────────
from tools import analyze_ecg_data

SYSTEM_PROMPT = (
    "You are a heart-health and mental-wellbeing assistant — a demo only, not a "
    "real doctor. "
    "When a user gives their age, blood pressure, and smoking status, call "
    "calculate_cardiac_risk for the point-based score; if they ask for the ML "
    "prediction, call predict_cardiac_risk_ml. When a user gives a list of "
    "numbers from a wearable or ECG device, call analyze_ecg_data. "
    "If the user wants a depression screen and provides nine 0–3 answers, call "
    "score_phq9; for an anxiety screen with seven 0–3 answers, call score_gad7. "
    "To gauge the emotional tone of a message, call analyze_mood. "
    "If the user uploads an ECG file the system will analyse it automatically "
    "and pass you the findings; explain them in clear, friendly language. "
    "Be warm and supportive when discussing mental health. Deterministic safety "
    "guardrails handle medical and self-harm emergencies before you are called, "
    "so never override them. "
    "Always state you are a demo and the user should see a real professional."
)

CHAT_MODEL = "gpt-4o-mini"

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_cardiac_risk",
            "description": (
                "Estimate cardiac risk from age, systolic BP, and smoking status "
                "using a point-based scoring model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "systolic_bp": {"type": "integer"},
                    "smokes": {"type": "boolean"},
                },
                "required": ["age", "systolic_bp", "smokes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_ecg_data",
            "description": (
                "Analyse a list of numeric ECG/pulse samples from a wearable device."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "signal_array": {
                        "type": "array",
                        "items": {"type": "number"},
                    }
                },
                "required": ["signal_array"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_cardiac_risk_ml",
            "description": (
                "Predict cardiac risk from age, systolic BP, and smoking status "
                "using a trained scikit-learn model (returns a probability)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {"type": "integer"},
                    "systolic_bp": {"type": "integer"},
                    "smokes": {"type": "boolean"},
                },
                "required": ["age", "systolic_bp", "smokes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_phq9",
            "description": (
                "Score a PHQ-9 depression screener. Provide nine integer answers "
                "(0–3 each, in question order)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 3},
                        "minItems": 9,
                        "maxItems": 9,
                    }
                },
                "required": ["answers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_gad7",
            "description": (
                "Score a GAD-7 anxiety screener. Provide seven integer answers "
                "(0–3 each, in question order)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "answers": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 3},
                        "minItems": 7,
                        "maxItems": 7,
                    }
                },
                "required": ["answers"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_mood",
            "description": (
                "Classify the emotional tone of a short text into a mood label "
                "(positive/calm/anxious/stressed/sad) using a trained model."
            ),
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]

TOOL_DISPATCH = {
    "calculate_cardiac_risk": lambda a: calculate_cardiac_risk(
        a["age"], a["systolic_bp"], a["smokes"]
    ),
    "analyze_ecg_data": lambda a: analyze_ecg_data(a["signal_array"]),
    "predict_cardiac_risk_ml": lambda a: predict_cardiac_risk_ml(
        a["age"], a["systolic_bp"], a["smokes"]
    ),
    "score_phq9": lambda a: score_phq9(a["answers"]),
    "score_gad7": lambda a: score_gad7(a["answers"]),
    "analyze_mood": lambda a: analyze_mood_text(a["text"]),
}


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
        except (KeyError, FileNotFoundError):
            api_key = None
    if not api_key:
        st.error("OPENAI_API_KEY is not set (env var or Streamlit secret).")
        st.stop()
    return OpenAI(api_key=api_key)


def run_tool_loop(client: OpenAI, messages: list) -> str:
    """Drive the LLM until it returns a text response with no pending tool calls."""
    while True:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = json.loads(tc.function.arguments)
                result = TOOL_DISPATCH[fn_name](fn_args)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
        else:
            return msg.content


def handle_ecg_upload(uploaded_file, client: OpenAI) -> None:
    """Convert an ECG file, run vision analysis, and inject findings into chat."""
    with st.spinner("Analysing ECG image…"):
        try:
            b64 = convert_file_to_image(uploaded_file)
        except ValueError as exc:
            st.error(str(exc))
            return

        try:
            findings = analyze_ecg_visual(b64, client=client)
        except Exception as exc:
            st.error(f"Vision analysis failed: {exc}")
            return

    findings_text = (
        f"ECG file analysed:\n"
        f"- Heart rate: {findings['heart_rate_bpm'] or 'not detected'} BPM\n"
        f"- Rhythm: {findings['rhythm_classification']}\n"
        f"- Observations: {findings['observations']}"
    )

    st.info(findings_text)

    # Inject findings as a user turn so the LLM can explain them
    prompt = (
        f"I just uploaded an ECG image. The automated analysis found:\n{findings_text}\n"
        "Can you explain what this means in plain language?"
    )
    st.session_state.messages.append({"role": "user", "content": prompt})

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]

    with st.spinner("Generating explanation…"):
        reply = run_tool_loop(client, api_messages)

    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})


def main() -> None:
    st.set_page_config(page_title="Cardiac Health PoC", page_icon="❤️")
    st.title("Cardiac Health Assistant (Demo)")
    st.caption("Not a medical device. Always consult a real doctor.")

    with st.sidebar:
        st.header("ECG File Upload")
        uploaded = st.file_uploader(
            "Upload an ECG image or PDF", type=["jpg", "jpeg", "png", "pdf"]
        )
        if uploaded and st.button("Analyse ECG"):
            client = get_client()
            handle_ecg_upload(uploaded, client)

        st.divider()
        st.header("Quick Risk Check")
        with st.form("risk_form"):
            age = st.number_input("Age", min_value=1, max_value=120, value=45)
            bp = st.number_input("Systolic BP (mmHg)", min_value=60, max_value=250, value=120)
            smokes = st.checkbox("I smoke")
            col_rule, col_ml = st.columns(2)
            with col_rule:
                rule_submit = st.form_submit_button("Point-based")
            with col_ml:
                ml_submit = st.form_submit_button("ML predict")
            if rule_submit:
                st.info(calculate_cardiac_risk(int(age), int(bp), smokes))
            if ml_submit:
                st.info(predict_cardiac_risk_ml(int(age), int(bp), smokes))

        st.divider()
        st.header("Depression Screener (PHQ-9)")
        with st.form("phq9_form"):
            phq9_answers = [
                st.select_slider(
                    q, options=[0, 1, 2, 3], value=0,
                    format_func=lambda v: LIKERT_LABELS[v], key=f"phq9_{i}",
                )
                for i, q in enumerate(PHQ9_QUESTIONS)
            ]
            if st.form_submit_button("Score PHQ-9"):
                st.info(score_phq9(phq9_answers))

        st.divider()
        st.header("Anxiety Screener (GAD-7)")
        with st.form("gad7_form"):
            gad7_answers = [
                st.select_slider(
                    q, options=[0, 1, 2, 3], value=0,
                    format_func=lambda v: LIKERT_LABELS[v], key=f"gad7_{i}",
                )
                for i, q in enumerate(GAD7_QUESTIONS)
            ]
            if st.form_submit_button("Score GAD-7"):
                st.info(score_gad7(gad7_answers))

        st.divider()
        st.header("Mood Tracker")
        if st.session_state.get("mood_history"):
            history = st.session_state.mood_history
            st.caption(f"Latest mood: {history[-1]['label']}")
            st.line_chart(
                {"confidence": [h["score"] for h in history]}
            )
        else:
            st.caption("Mood trend appears here as you chat.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "mood_history" not in st.session_state:
        st.session_state.mood_history = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_input = st.chat_input("Ask about heart health…")
    if not user_input:
        return

    # Phase 2 — emergency intercept: bypass LLM entirely
    if check_emergency_triggers(user_input):
        with st.chat_message("assistant"):
            st.error(EMERGENCY_MESSAGE)
        st.session_state.messages.append(
            {"role": "assistant", "content": f"🚨 {EMERGENCY_MESSAGE}"}
        )
        return

    # Mental-health crisis intercept: also runs before any LLM call
    if check_crisis_triggers(user_input):
        with st.chat_message("assistant"):
            st.error(CRISIS_MESSAGE)
        st.session_state.messages.append(
            {"role": "assistant", "content": CRISIS_MESSAGE}
        )
        return

    # Mood tracking — record the emotional tone of this turn for the sidebar trend
    mood = analyze_mood(user_input)
    st.session_state.mood_history.append(mood)

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ]

    client = get_client()
    with st.spinner("Thinking…"):
        reply = run_tool_loop(client, api_messages)

    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
