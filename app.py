import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import SUBJECTS
from rag import get_chain
from uploader import ingest_uploaded_file

st.set_page_config(page_title="AI Tutor", layout="wide")
st.title("Your Personal AI Tutor")

# ── sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📚 Subject")
    selected_subject = st.selectbox(
        "Choose subject",
        SUBJECTS,
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 📂 Upload material")
    upload_subject = st.selectbox(
        "File belongs to",
        [s for s in SUBJECTS if s != "All Subjects"],
    )
    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded:
        if st.button("Add to knowledge base", use_container_width=True):
            for file in uploaded:
                count = ingest_uploaded_file(file, subject=upload_subject)
                st.success(f"✓ {file.name} — {count} chunks")

# ── per-subject chat history ──────────────────────────────────
chat_key = f"messages_{selected_subject}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = []

st.markdown(f"#### 💬 {selected_subject}")

for message in st.session_state[chat_key]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── chat input ────────────────────────────────────────────────
if user_input := st.chat_input(f"Ask about {selected_subject}..."):
    st.session_state[chat_key].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    chain = get_chain(selected_subject)

    with st.chat_message("assistant"):
        response = chain.stream(user_input)
        full_response = st.write_stream(response)

    st.session_state[chat_key].append({"role": "assistant", "content": full_response})