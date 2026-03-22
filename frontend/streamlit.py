import logging

import requests
import streamlit as st

from config import FrontendSettings
from logger_config import setup_logging

frontend_settings = FrontendSettings()
FASTAPI_BACKEND_URL = frontend_settings.FASTAPI_BACKEND_URL

API_CHAT_URL = f"{FASTAPI_BACKEND_URL}/chat"
API_FEEDBACK_URL = f"{FASTAPI_BACKEND_URL}/feedback"
API_MODELS_URL = f"{FASTAPI_BACKEND_URL}/models"

st.set_page_config(page_title="WIKI RAG - Chat", layout="wide")

setup_logging("frontend")
logger = logging.getLogger(__name__)

# CSS
st.markdown(
    """
    <style>
    /* Suggested prompt buttons */
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        border: 1px solid rgba(49, 51, 63, 0.2);
        background: rgba(240, 242, 246, 0.8);
        text-align: left;
        white-space: normal;
        line-height: 1.2rem;
    }
    div.stButton > button:hover {
        border-color: rgba(49, 51, 63, 0.45);
        background: rgba(240, 242, 246, 1.0);
    }

    /* Dark text in disabled text area (chat history) */
    .stTextArea textarea[disabled] {
        color: #000 !important;
        -webkit-text-fill-color: #000 !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- state ---
# chat conversation
if "history" not in st.session_state:
    st.session_state.history = []

if "model_name" not in st.session_state:
    st.session_state.model_name = "llama3.2"

if "suggested_prompts" not in st.session_state:
    st.session_state.suggested_prompts = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "pending_source" not in st.session_state:
    st.session_state.pending_source = None

if "feedback_available" not in st.session_state:
    st.session_state.feedback_available = False

if "feedback_sent" not in st.session_state:
    st.session_state.feedback_sent = False

if "http" not in st.session_state:
    st.session_state.http = requests.Session()


@st.cache_data(ttl=20)
def fetch_available_models():
    try:
        r = requests.get(API_MODELS_URL, timeout=5)
        r.raise_for_status()
        models = r.json().get("models", ["llama3.2"])
        logger.info(f"Avaialbe ollama models: {models}")
        return models
    except Exception as e:
        logger.exception(f"Error while downloading models: {e}")
        st.sidebar.error(f"Error while downloading models: {e}")
        return ["llama3.2"]


available_models = fetch_available_models()

if st.session_state.model_name not in available_models:
    st.session_state.model_name = available_models[0]


def render_history_text(history):
    """Render lines into text box"""
    lines = []
    for msg in history:
        who = "Ty" if msg["role"] == "user" else "LLM"
        lines.append(f"{who}: {msg['content']}")
    return "\n\n".join(lines)


def clear_chat():
    """Clear all states"""
    st.session_state.history = []
    st.session_state.suggested_prompts = []
    st.session_state.pending_question = None
    st.session_state.pending_source = None
    st.session_state.feedback_available = False
    st.session_state.feedback_sent = False


def call_chat_api(question: str, model_name: str) -> dict:
    """Call chat endpoint"""
    payload = {"question": question, "model_name": model_name}
    r = st.session_state.http.post(API_CHAT_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def call_feedback_api(rating: str) -> dict:
    """Call feedback endpoint"""
    payload = {"rating": rating}
    r = st.session_state.http.post(API_FEEDBACK_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def queue_question(question: str, source: str):
    q = (question or "").strip()
    if not q:
        return

    # reset UI pieces for new question
    st.session_state.suggested_prompts = []
    st.session_state.feedback_available = False
    st.session_state.feedback_sent = False

    st.session_state.history.append({"role": "user", "content": q})

    st.session_state.pending_question = q
    st.session_state.pending_source = source

    st.rerun()


def process_pending_question_if_any():
    """GUI engine"""
    q = st.session_state.pending_question
    if not q:
        return

    st.session_state.pending_question = None
    st.session_state.pending_source = None

    try:
        data = call_chat_api(q, st.session_state.model_name)
        answer = data.get("answer", "")
        suggested = data.get("suggested_prompts", []) or []

        st.session_state.history.append({"role": "assistant", "content": answer})

        if len(suggested) > 0:
            st.session_state.suggested_prompts = suggested

        # po udanej odpowiedzi z /chat -> feedback można wysłać
        st.session_state.feedback_available = True
        st.session_state.feedback_sent = False

    except requests.RequestException as e:
        st.session_state.history.append(
            {"role": "assistant", "content": f"Błąd wywołania API: {e}"}
        )
        st.session_state.feedback_available = False
        st.session_state.feedback_sent = False

    st.rerun()


def send_feedback(rating: str):
    try:
        call_feedback_api(rating)
    except requests.RequestException as e:
        st.session_state.history.append(
            {"role": "assistant", "content": f"Błąd wysyłania feedbacku: {e}"}
        )
    finally:
        st.session_state.feedback_available = False
        st.session_state.feedback_sent = True
        st.rerun()


# LAYOUT
# --- top bar ---
left, right = st.columns([1, 5])
with left:
    st.button("Wyczyść czat", on_click=clear_chat, use_container_width=True)

with right:
    st.selectbox(
        "Model",
        options=available_models,
        key="model_name",
    )

# --- history ---
st.text_area(
    "Historia czatu",
    value=render_history_text(st.session_state.history),
    height=370,
    disabled=True,
)

# --- suggested prompts + feedback (two columns) ---
st.markdown("**Sugerowane pytania**")

left, right = st.columns([5, 1], vertical_alignment="top")

with left:
    suggested = st.session_state.suggested_prompts
    if not suggested:
        st.text_input(
            label="Suggested prompts (puste)",
            value="",
            placeholder="Brak sugestii",
            disabled=True,
            label_visibility="collapsed",
        )
    else:
        cols_per_row = 3
        for i in range(0, len(suggested), cols_per_row):
            row = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx >= len(suggested):
                    break
                prompt = suggested[idx]
                with row[j]:
                    st.button(
                        prompt,
                        key=f"suggest_{idx}_{prompt}",
                        on_click=queue_question,
                        args=(prompt, "suggested"),
                        use_container_width=True,
                    )

with right:
    with st.container(border=True):
        st.markdown(
            "<h5 style='text-align:center; margin:0;'>Feedback</h5>",
            unsafe_allow_html=True,
        )

        feedback_disabled = (
            bool(st.session_state.pending_question)
            or (not st.session_state.feedback_available)
            or st.session_state.feedback_sent
        )

        b1, b2 = st.columns(2)
        with b1:
            st.button(
                "Up",
                use_container_width=True,
                disabled=feedback_disabled,
                on_click=send_feedback,
                args=("up",),
            )
        with b2:
            st.button(
                "Down",
                use_container_width=True,
                disabled=feedback_disabled,
                on_click=send_feedback,
                args=("down",),
            )

with st.form("chat_form", clear_on_submit=True):
    i1, i2 = st.columns([8, 1], vertical_alignment="bottom")

    with i1:
        user_text = st.text_input(
            "Twoja wiadomość",
            placeholder="Napisz pytanie i naciśnij Enter lub przycisk OK...",
            label_visibility="collapsed",
            disabled=bool(st.session_state.pending_question),
        )

    with i2:
        submitted = st.form_submit_button(
            "OK",
            use_container_width=True,
            disabled=bool(st.session_state.pending_question),
        )

if submitted:
    queue_question(user_text, "user_input")

process_pending_question_if_any()
