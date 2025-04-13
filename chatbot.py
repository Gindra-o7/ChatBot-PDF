import os
import streamlit as st
import requests
import PyPDF2
from dotenv import load_dotenv
from io import BytesIO
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize embedding model
EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Konfigurasi halaman
st.set_page_config(page_title="AI Comparison", layout="wide")

# Custom CSS untuk tampilan yang sesuai dengan gambar
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid #ddd;
        margin-bottom: 20px;
    }
    .header h1 {
        margin: 0;
        font-family: monospace;
        font-weight: bold;
    }
    .header span {
        font-family: monospace;
        color: #666;
    }
    .upload-section {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
    }
    .model-dropdown {
        background-color: #e8f8e8;
        border-radius: 20px;
        padding: 8px 15px;
        margin-bottom: 15px;
        text-align: center;
    }
    .versus-button {
        background-color: #ff0000;
        border-radius: 50px;
        padding: 15px 25px;
        margin: 10px auto;
        width: 100px;
        text-align: center;
    }
    .upload-button {
        background-color: #fff9e6;
        border-radius: 20px;
        padding: 12px;
        width: 100%;
        text-align: center;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    /* Adjust input fields */
    div[data-baseweb="select"] > div {
        border-radius: 20px;
    }
    div[data-baseweb="base-input"] > div {
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>AI Comparison</h1>
    <span>v1.32-alpha</span>
</div>
""", unsafe_allow_html=True)

# Fungsi untuk membersihkan teks
def clean_text(text):
    """
    Membersihkan teks dari karakter dan format yang tidak diinginkan
    """
    # Menghapus semua format \boxed{...}
    cleaned_text = re.sub(r'\\boxed\{.*?\}', '', text)
    # Menghapus tanda ``` di awal dan akhir
    cleaned_text = cleaned_text.replace("```", "")
    # Menghilangkan karakter non-ASCII
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', cleaned_text)
    # Menghapus whitespace berlebih
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text

# Fungsi untuk mengekstrak dan memproses teks dari PDF
def process_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        full_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += clean_text(page_text) + "\n"
        
        # Menggunakan text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=100,
            length_function=len
        )
        chunks = text_splitter.split_text(full_text)
        
        # Buat embeddings dan indeks FAISS
        if chunks:
            embeddings = EMBEDDING_MODEL.encode(chunks)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))
            
            return chunks, index
        return [], None
    except Exception as e:
        st.error(f"Error memproses PDF: {e}")
        return [], None

# Fungsi untuk mendapatkan konteks yang relevan berdasarkan query
def get_relevant_context(query, chunks, index, top_k=3):
    if not chunks or index is None:
        return ""
    
    clean_query = clean_text(query)
    query_embedding = EMBEDDING_MODEL.encode([clean_query])
    _, indices = index.search(query_embedding.astype('float32'), top_k)
    
    # Mengambil chunk teks yang paling relevan
    relevant_context = "\n".join([chunks[i] for i in indices[0] if i < len(chunks)])
    print(relevant_context)
    return relevant_context

# Fungsi untuk memanggil API OpenRouter
def call_openrouter_api(prompt, model_name, context=""):
    url = f"{BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Jika ada konteks yang relevan, tambahkan ke prompt
    if context:
        full_prompt = f"Konteks dari dokumen:\n{context}\n\nPertanyaan: {prompt}"
    else:
        full_prompt = prompt
    
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": full_prompt}],
    }
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Memastikan tidak ada error HTTP
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error saat memanggil API OpenRouter: {e}")
        return "Terjadi kesalahan saat menghubungi model."

# Inisialisasi session state
if "response1" not in st.session_state:
    st.session_state.response1 = "Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet."
if "response2" not in st.session_state:
    st.session_state.response2 = "Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet. Lorem ipsum dolor sit amet."
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = []
if "pdf_index" not in st.session_state:
    st.session_state.pdf_index = None

# Daftar model (ganti dengan model gratis yang tersedia di OpenRouter)
model_list = [
    "meta-llama/llama-3-8b-instruct", 
    "mistralai/mixtral-8x7b-instruct", 
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
    "deepseek/deepseek-r1-zero:free",
]

# Upload dan input section
col1, col2, col3 = st.columns([1, 4, 1])

with col1:
    # Custom upload button with styling
    st.markdown('<div class="upload-button">', unsafe_allow_html=True)
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process PDF when uploaded
    if pdf_file is not None and pdf_file != st.session_state.get('last_uploaded_file'):
        with st.spinner("Memproses dokumen PDF..."):
            st.session_state.pdf_chunks, st.session_state.pdf_index = process_pdf(pdf_file)
            st.session_state.last_uploaded_file = pdf_file
            if st.session_state.pdf_chunks:
                st.success(f"Berhasil memproses PDF menjadi {len(st.session_state.pdf_chunks)} chunks")

with col2:
    user_prompt = st.text_input("", placeholder="Tanyakan disini...", label_visibility="collapsed")

with col3:
    # Custom send button with styling
    st.markdown('<div class="send-button">', unsafe_allow_html=True)
    send_button = st.button("Send", key="send")
    st.markdown('</div>', unsafe_allow_html=True)

# Model response sections
col1, mid_col, col2 = st.columns([5, 1, 5])

with col1:
    # Player 1 model dropdown with custom styling
    st.markdown('<div class="model-dropdown">', unsafe_allow_html=True)
    model1 = st.selectbox("", model_list, key="model1", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Response display for Player 1
    st.markdown('<div class="response-box">', unsafe_allow_html=True)
    st.write(st.session_state.response1)
    st.markdown('</div>', unsafe_allow_html=True)

# Versus button in middle column
with mid_col:
    st.markdown('<div class="versus-button">Versus</div>', unsafe_allow_html=True)

with col2:
    # Player 2 model dropdown with custom styling
    st.markdown('<div class="model-dropdown">', unsafe_allow_html=True)
    model2 = st.selectbox("", model_list, index=1, key="model2", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Response display for Player 2
    st.markdown('<div class="response-box">', unsafe_allow_html=True)
    st.write(st.session_state.response2)
    st.markdown('</div>', unsafe_allow_html=True)

# Bottom section for Comparator model and Compare button
col1, col2 = st.columns([1, 1])

with col1:
    # Comparator model dropdown with custom styling
    comparator_model = st.selectbox("", model_list, key="comparator", label_visibility="collapsed")

with col2:
    # Compare button with custom styling
    st.markdown('<div class="compare-button">', unsafe_allow_html=True)
    compare_button = st.button("Tap to Compare those response...", key="compare")
    st.markdown('</div>', unsafe_allow_html=True)

# Debug expander (optional)
with st.expander("Debug Information", expanded=False):
    st.checkbox("Show PDF Processing Details", key="show_pdf_details")
    if st.session_state.get("show_pdf_details", False):
        if st.session_state.pdf_chunks:
            st.write(f"Jumlah chunks: {len(st.session_state.pdf_chunks)}")
            st.write("Contoh chunk pertama:")
            st.code(st.session_state.pdf_chunks[0][:500] + "..." if len(st.session_state.pdf_chunks[0]) > 500 else st.session_state.pdf_chunks[0])

# Process "Send" button logic
if send_button:
    if user_prompt:
        # Get relevant context from PDF if available
        relevant_context = ""
        if st.session_state.pdf_chunks and st.session_state.pdf_index:
            relevant_context = get_relevant_context(user_prompt, st.session_state.pdf_chunks, st.session_state.pdf_index)
        
        # Call API for both models
        with st.spinner(f"Memproses {model1}..."):
            st.session_state.response1 = call_openrouter_api(user_prompt, model1, relevant_context)
        
        with st.spinner(f"Memproses {model2}..."):
            st.session_state.response2 = call_openrouter_api(user_prompt, model2, relevant_context)
        
        # Force refresh
        st.rerun()
    else:
        st.warning("Silakan masukkan pertanyaan atau prompt terlebih dahulu.")

# Process "Compare" button logic
if compare_button:
    if st.session_state.response1 and st.session_state.response2:
        with st.spinner("Membandingkan hasil..."):
            comparison_prompt = f"""
            Bandingkan dua jawaban berikut dan tentukan mana yang lebih baik:
            
            Pertanyaan: {user_prompt}
            
            Jawaban Model 1 ({model1}):
            {st.session_state.response1}
            
            Jawaban Model 2 ({model2}):
            {st.session_state.response2}
            
            Analisis kedua jawaban tersebut berdasarkan: akurasi, kejelasan, kelengkapan, dan kegunaan.
            Berikan penjelasan detail mengapa satu jawaban lebih baik dari yang lain.
            """
            
            comparison_result = call_openrouter_api(comparison_prompt, comparator_model)
            
            st.markdown("### Hasil Perbandingan")
            st.write(comparison_result)
    else:
        st.warning("Silakan kirimkan pertanyaan terlebih dahulu untuk mendapatkan jawaban dari kedua model.")

# Informasi tambahan
with st.expander("Cara Menggunakan"):
    st.write("""
    1. **Upload PDF** (opsional) - Gunakan tombol "Upload PDF" untuk memberikan konteks tambahan dari dokumen PDF. Dokumen akan diproses menjadi chunk kecil untuk pencarian yang lebih akurat.
    2. **Masukkan Pertanyaan** - Ketik pertanyaan atau prompt di kotak teks.
    3. **Pilih Model** - Pilih model AI yang ingin dibandingkan (Player 1 dan Player 2).
    4. **Tekan Send** - Kirim pertanyaan untuk mendapatkan jawaban dari kedua model. Jika PDF telah diupload, sistem akan mencari bagian dokumen yang paling relevan dengan pertanyaan Anda.
    5. **Bandingkan Hasil** - Gunakan model komparator untuk mengevaluasi mana jawaban yang lebih baik.
    
    Pastikan Anda telah mengatur API key OpenRouter di file .env Anda.
    """)