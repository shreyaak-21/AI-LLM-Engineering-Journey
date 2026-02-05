# app.py
import streamlit as st
import json

from llm_client import generate_response
from prompt_builder import build_prompt
from tokenizer_utils import count_tokens
from chunking_utils import chunk_text
from json_utils import is_valid_json, extract_json, repair_json
from experiment_tracker import log_experiment

# -----------------------------
# Configuration
# -----------------------------
MAX_CONTEXT_TOKENS = 2048

st.set_page_config(page_title="LLM Behavior Analyzer", layout="wide")
st.title("🧠 LLM Behavior Analyzer & Chat Playground")

# -----------------------------
# Utility Functions
# -----------------------------
def trim_chat_history(system_prompt, history, max_tokens):
    trimmed = []
    total_text = system_prompt

    for turn in reversed(history):
        candidate = (
            f"User: {turn['user']}\n"
            f"Assistant: {turn['assistant']}\n"
            + total_text
        )
        if count_tokens(candidate) < max_tokens:
            trimmed.insert(0, turn)
            total_text = candidate
        else:
            break

    return trimmed

# -----------------------------
# Session State
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Prompt Controls
# -----------------------------
st.subheader("🔧 Prompt Controls")

system_prompt = st.text_area(
    "System Prompt",
    "You are a helpful AI assistant.",
    height=100
)

user_prompt = st.text_area("User Prompt", height=200)

temperature = st.slider(
    "Temperature (Creativity Level)",
    0.1, 1.0, 0.7, 0.1
)

response_format = st.selectbox(
    "Response Format",
    ["Paragraph", "Bullet Points", "JSON"]
)

# -----------------------------
# Generate Response
# -----------------------------
st.markdown("---")
st.subheader("💬 Generate Response")

if st.button("Generate Response") and user_prompt.strip():

    # 🔹 Format conditioning
    format_instruction = ""
    if response_format == "Bullet Points":
        format_instruction = "Respond using clear bullet points."
    elif response_format == "JSON":
        format_instruction = (
            "Respond strictly in valid JSON format. "
            "Do not include explanations outside JSON."
        )

    final_system_prompt = system_prompt + "\n\n" + format_instruction

    # 🔹 Build conversation
    conversation = ""
    for turn in st.session_state.chat_history:
        conversation += (
            f"User: {turn['user']}\n"
            f"Assistant: {turn['assistant']}\n"
        )

    conversation += f"User: {user_prompt}\nAssistant:"
    full_prompt = build_prompt(final_system_prompt, conversation)

    token_count = count_tokens(full_prompt)
    st.info(f"🧮 Tokens used: {token_count} / {MAX_CONTEXT_TOKENS}")

    # 🔹 Inference
    if token_count > MAX_CONTEXT_TOKENS:
        st.warning("⚠️ Context exceeded. Applying chunking.")
        chunks = chunk_text(user_prompt)
        responses = []

        for idx, chunk in enumerate(chunks):
            st.write(f"🔹 Chunk {idx + 1}/{len(chunks)}")
            chunk_prompt = build_prompt(final_system_prompt, chunk)
            resp = generate_response(chunk_prompt, temperature)

            if response_format == "JSON":
                resp = extract_json(resp)

            responses.append(resp)

        final_response = "\n\n".join(responses)

    else:
        raw_response = generate_response(full_prompt, temperature)
        final_response = raw_response

        # 🧪 JSON validation
        if response_format == "JSON":
            cleaned = extract_json(raw_response)

            if not is_valid_json(cleaned):
                st.warning("⚠️ Invalid JSON detected. Repairing...")
                repaired = repair_json(
                    lambda p, temperature=0.0: generate_response(p, temperature),
                    cleaned
                )
                repaired_clean = extract_json(repaired)

                if is_valid_json(repaired_clean):
                    final_response = repaired_clean
                    st.success("✅ JSON repaired")
                else:
                    st.error("❌ JSON repair failed")

            else:
                final_response = cleaned

    # 🔹 Save history
    st.session_state.chat_history.append({
        "user": user_prompt,
        "assistant": final_response
    })

    st.session_state.chat_history = trim_chat_history(
        final_system_prompt,
        st.session_state.chat_history,
        MAX_CONTEXT_TOKENS
    )

    # 🔹 Log experiment
    log_experiment(
        system_prompt=final_system_prompt,
        user_prompt=user_prompt,
        full_prompt=full_prompt,
        temperature=temperature,
        response_format=response_format,
        model="mistral (ollama)",
        response=final_response,
        token_count=token_count
    )

    st.markdown("### 🤖 Final Response")
    st.write(final_response)

# -----------------------------
# Behavior Comparison Mode
# -----------------------------
st.markdown("---")
st.subheader("🔬 Behavior Comparison Mode")

if st.button("Compare Same Prompt at Different Temperatures") and user_prompt.strip():
    temps = [0.2, 0.7, 1.0]
    cols = st.columns(3)

    for col, temp in zip(cols, temps):
        with col:
            st.markdown(f"### 🌡️ Temp = {temp}")
            prompt = build_prompt(system_prompt, user_prompt)
            st.write(generate_response(prompt, temp))

# -----------------------------
# Chat History
# -----------------------------
st.markdown("---")
st.subheader("🗨️ Chat History")

for turn in st.session_state.chat_history:
    st.markdown(f"**🧑 User:** {turn['user']}")
    st.markdown(f"**🤖 Assistant:** {turn['assistant']}")
    st.markdown("---")

# -----------------------------
# Experiment Log
# -----------------------------
st.subheader("📊 Experiment Log")

if st.checkbox("Show experiment history"):
    try:
        with open("experiments.jsonl", "r", encoding="utf-8") as f:
            for line in reversed(f.readlines()[-10:]):
                exp = json.loads(line)
                st.code(exp["response"])
    except FileNotFoundError:
        st.info("No experiments logged yet.")
