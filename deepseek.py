import os
import PyPDF2
import requests
import streamlit as st
from dotenv import load_dotenv

# Muat variabel lingkungan dari file .env
load_dotenv()
BASE_URL = os.getenv("OPENROUTER_BASE_URL")
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Fungsi untuk membaca teks dari PDF
def extract_text_from_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Fungsi untuk mengirim permintaan ke OpenRouter API
def query_openrouter(prompt):
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek/deepseek-r1-zero:free",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            return "Error: No choices found in the response."
    else:
        return f"Error: {response.status_code}, {response.text}"

# Fungsi untuk menghapus riwayat chat
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

# Antarmuka Streamlit
st.set_page_config(page_title="DeepSeek PDF Chatbot", page_icon="🤖")

with st.sidebar:
    st.title("Menu:")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = extract_text_from_pdf(pdf_docs)
            st.session_state["pdf_text"] = raw_text
            st.success("Done")
    st.button("Clear Chat History", on_click=clear_chat_history)

st.title("Chat with PDF files using DeepSeek🤖")
st.write("Welcome to the chat!")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if "pdf_text" in st.session_state:
        full_prompt = f"Dokumen PDF:\n{st.session_state['pdf_text']}\n\nPertanyaan: {prompt}\nJawaban:"
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_openrouter(full_prompt)
                st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Silakan unggah file PDF terlebih dahulu.")