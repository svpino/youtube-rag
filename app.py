import streamlit as st
from document_processor import DocumentProcessor
from embedding_indexer import EmbeddingIndexer
from rag_chain import RAGChain
from chatbot import Chatbot

import os
from dotenv import load_dotenv
    

@st.cache_resource
def initialize_chatbot(file_path):
    
    load_dotenv()

    GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
    processor = DocumentProcessor(file_path)
    texts = processor.load_and_split()
    indexer = EmbeddingIndexer(GOOGLE_API_KEY)
    vectorstore = indexer.create_vectorstore(texts)
    rag_chain = RAGChain(GOOGLE_API_KEY, vectorstore)
    
    return Chatbot(rag_chain.create_chain())

st.title("RAG Chatbot")

uploaded_file = st.file_uploader("Upload a text file for the knowledge base", type="txt")

if uploaded_file:
    with open("transcription.txt", "ab") as f:
        f.write(uploaded_file.getbuffer())

    chatbot = initialize_chatbot("transcription.txt")

chatbot = initialize_chatbot("transcription.txt")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt := st.chat_input("Ask a question"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chatbot.get_response(prompt)
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})