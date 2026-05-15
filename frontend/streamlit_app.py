# frontend/streamlit_app.py

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="AI Research Assistant", page_icon="📚", layout="wide")

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.title("📂 Upload Documents")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:

        files = {"file": uploaded_file}

        with st.spinner("Uploading and indexing PDF..."):

            response = requests.post(f"{API_URL}/upload", files=files)

        if response.status_code == 200:

            st.success(response.json()["message"])

        else:

            st.error("Upload failed")


# =========================
# MAIN TITLE
# =========================

st.title("📚 AI Research Assistant")

st.markdown("Ask intelligent questions across uploaded research documents using RAG.")

st.info("""
👋 Welcome to the AI Research Assistant!

Upload research papers, technical documents, or PDFs and ask intelligent questions across them using Retrieval-Augmented Generation (RAG).

### 🚀 Features
- Multi-document semantic search
- AI-powered contextual answers
- ChromaDB vector retrieval
- OpenAI grounded generation
- Conversational document querying

Start by uploading a PDF from the sidebar.
""")

# =========================
# SESSION CHAT HISTORY
# =========================

if "messages" not in st.session_state:

    st.session_state.messages = []

# =========================
# DISPLAY OLD MESSAGES
# =========================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================
# CHAT INPUT
# =========================

question = st.chat_input("Ask something about the uploaded document...")

# =========================
# HANDLE USER QUESTION
# =========================

if question:

    # USER MESSAGE

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):

        st.markdown(question)

    # AI RESPONSE

    with st.chat_message("assistant"):

        with st.spinner("Researching documents..."):

            response = requests.post(f"{API_URL}/query", json={"question": question})

            data = response.json()

            answer = data["answer"]

            sources = data.get("sources", [])

            st.markdown(answer)

            # DISPLAY SOURCES

            if sources:

                st.markdown("---")
                st.markdown("## 📚 Source References")

                for idx, source in enumerate(sources, start=1):

                    with st.expander(f"Source {idx}"):

                        st.markdown(f"**Document:** {source['source']}")

                        st.markdown(f"**Page:** {source['page']}")

                        st.markdown("**Retrieved Context:**")

                        st.write(source["content"])

    # SAVE ASSISTANT MESSAGE

    st.session_state.messages.append({"role": "assistant", "content": answer})
