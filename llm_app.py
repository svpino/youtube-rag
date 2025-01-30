import streamlit as st
from llm_chain import SimpleLLM
import os
from dotenv import load_dotenv

@st.cache_resource
def initialize_chatbot():
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    return SimpleLLM(GOOGLE_API_KEY).create_chain()

st.title("Simple LLM Chatbot")

chatbot = initialize_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chatbot.invoke({"question": prompt})
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})