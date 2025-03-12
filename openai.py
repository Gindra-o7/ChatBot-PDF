import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Kelas untuk memproses dokumen PDF
class PDFProcessor:
    def __init__(self, pdf_docs):
        self.pdf_docs = pdf_docs

    # Mengambil keseluruhan text pada PDF
    def get_text(self):
        text = ""
        for pdf in self.pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    # Membagi teks menjadi beberapa section
    def get_text_chunks(self):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(self.get_text())
        return chunks


# Kelas untuk membuat vector store
class VectorStore:
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def create_vectorstore(self):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=self.text_chunks, embedding=embeddings)
        return vectorstore


# Kelas untuk membuat conversation chain
class ConversationChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def create_conversation_chain(self):
        llm = ChatOpenAI(model_name="gpt-4")

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)

        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=self.vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain


# Fungsi untuk menangani pertanyaan pengguna
def handle_userinput(user_question):
    if st.session_state.conversation is not None:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    else:
        st.error("Conversation chain is not initialized. Please upload PDFs first.")


# Fungsi utama untuk menjalankan aplikasi Streamlit
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True, )
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get pdf text
                pdf_processor = PDFProcessor(pdf_docs)
                raw_text = pdf_processor.get_text()

                # Get the text chunks
                text_chunks = pdf_processor.get_text_chunks()

                # Create vector store
                vectorstore = VectorStore(text_chunks).create_vectorstore()

                # Create conversation chain
                st.session_state.conversation = ConversationChain(vectorstore).create_conversation_chain()

if __name__ == '__main__':
    main()