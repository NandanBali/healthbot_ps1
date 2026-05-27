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
from app.tools.ecg_vision_tool import analyze_ecg_visual
from app.tools.risk_tool import calculate_cardiac_risk
from app.utils.file_processor import convert_file_to_image

# ── Legacy numeric-array ECG tool (Phase 1) ───────────────────────────────────
from tools import analyze_ecg_data

SYSTEM_PROMPT = (
    "You are a heart health assistant — a demo only, not a real doctor. "
    "When a user gives their age, blood pressure, and smoking status, call "
    "calculate_cardiac_risk. When a user gives a list of numbers from a "
    "wearable or ECG device, call analyze_ecg_data. "
    "If the user uploads an ECG file the system will analyse it automatically "
    "and pass you the findings; explain them in clear, friendly language. "
    "Always state you are a demo and the user should see a real doctor."
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
]

TOOL_DISPATCH = {
    "calculate_cardiac_risk": lambda a: calculate_cardiac_risk(
        a["age"], a["systolic_bp"], a["smokes"]
    ),
    "analyze_ecg_data": lambda a: analyze_ecg_data(a["signal_array"]),
}


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY environment variable is not set.")
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
            if st.form_submit_button("Calculate Risk"):
                result = calculate_cardiac_risk(int(age), int(bp), smokes)
                st.info(result)

    if "messages" not in st.session_state:
        st.session_state.messages = []

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
