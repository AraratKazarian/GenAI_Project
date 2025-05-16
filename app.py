# app.py
import sys

# Quick patch to prevent torch.classes inspection crash on Windows
sys.modules["torch.classes"] = None

import streamlit as st
from rag_backend import get_rag_answer

st.set_page_config(page_title="RAG Chat", layout="centered")

st.title("🧠 RAG Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.chat_input("Ask something...")
if user_input:
    # Add user message
    st.session_state.chat_history.append(("user", user_input))

    # Get RAG-based answer
    answer = get_rag_answer(user_input)

    # Add bot message
    st.session_state.chat_history.append(("bot", answer))

# Render chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message)
    else:
        with st.chat_message("assistant"):
            st.markdown(message)
