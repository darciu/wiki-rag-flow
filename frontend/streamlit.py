import requests
import streamlit as st

API_CHAT_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="WIKI RAG - Chat", layout="wide")

# --- CSS (kafelki suggested prompts) ---
st.markdown(
    """
    <style>
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
    </style>
    """,
    unsafe_allow_html=True,
)

# --- state ---
if "history" not in st.session_state:
    st.session_state.history = []   # [{"role": "user"/"assistant", "content": "..."}]

if "model_name" not in st.session_state:
    st.session_state.model_name = "llama3.2"

if "suggested_prompts" not in st.session_state:
    st.session_state.suggested_prompts = []  # list[str]

# pending flow: najpierw pokaż pytanie, potem dopiero dobij odpowiedź
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

if "pending_source" not in st.session_state:
    st.session_state.pending_source = None  # "user_input" | "suggested"


# --- helpers ---
def render_history_text(history):
    lines = []
    for msg in history:
        who = "Ty" if msg["role"] == "user" else "LLM"
        lines.append(f"{who}: {msg['content']}")
    return "\n\n".join(lines)


def clear_chat():
    st.session_state.history = []
    st.session_state.suggested_prompts = []
    st.session_state.pending_question = None
    st.session_state.pending_source = None


def call_chat_api(question: str, model_name: str) -> dict:
    payload = {"question": question, "model_name": model_name}
    r = requests.post(API_CHAT_URL, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def queue_question(question: str, source: str):
    """Dodaje pytanie do historii natychmiast i ustawia je jako pending do obróbki w kolejnym rerunie."""
    q = (question or "").strip()
    if not q:
        return

    # wymaganie: przy manualnym pytaniu i przy kliknięciu sugestii czyścimy suggested_prompts
    st.session_state.suggested_prompts = []

    # dopisz pytanie od razu
    st.session_state.history.append({"role": "user", "content": q})

    # ustaw pending (odpowiedź dociągniemy w kolejnym przebiegu)
    st.session_state.pending_question = q
    st.session_state.pending_source = source

    st.rerun()


def process_pending_question_if_any():
    """Jeśli jest pending_question, to robi request do /chat i dopisuje odpowiedź + suggested_prompts."""
    q = st.session_state.pending_question
    if not q:
        return

    # zdejmij pending od razu, żeby nie zapętlić się przy ewentualnych wyjątkach
    st.session_state.pending_question = None
    st.session_state.pending_source = None

    try:
        data = call_chat_api(q, st.session_state.model_name)
        answer = data.get("answer", "")
        route = data.get("route", "")
        suggested = data.get("suggested_prompts", []) or []

        extra = f"\n\n[route: {route}]" if route else ""
        st.session_state.history.append({"role": "assistant", "content": answer})

        # wymaganie: jeśli endpoint zwróci pustą listę -> nie uzupełniaj pola
        if len(suggested) > 0:
            st.session_state.suggested_prompts = suggested

    except requests.RequestException as e:
        st.session_state.history.append(
            {"role": "assistant", "content": f"Błąd wywołania API: {e}"}
        )

    st.rerun()


# --- UI top bar ---
c1, c2 = st.columns([1, 3])
with c1:
    st.button("Wyczyść czat", on_click=clear_chat, use_container_width=True)

with c2:
    st.selectbox(
        "Model",
        options=["llama3.2", "llama3", "mistral", "qwen2.5"],
        key="model_name",
    )

# --- history (read-only) ---
st.text_area(
    "Historia czatu",
    value=render_history_text(st.session_state.history),
    height=520,
    disabled=True,
)

# --- suggested prompts between history and input ---
st.markdown("**Suggested prompts**")
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

# --- input + submit ---
with st.form("chat_form", clear_on_submit=True):
    user_text = st.text_input(
        "Twoja wiadomość",
        placeholder="Napisz pytanie i naciśnij Enter lub OK...",
        label_visibility="collapsed",
        disabled=bool(st.session_state.pending_question),  # opcjonalnie: blokuj, gdy w toku
    )
    submitted = st.form_submit_button(
        "OK",
        use_container_width=True,
        disabled=bool(st.session_state.pending_question),  # opcjonalnie: blokuj, gdy w toku
    )

if submitted:
    queue_question(user_text, "user_input")

# --- worker: jeśli jest pending -> pobierz odpowiedź ---
process_pending_question_if_any()