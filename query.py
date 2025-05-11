import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# 🔐 Lade Umgebungsvariablen
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# 📄 Dokument laden und in Chunks teilen
def load_and_split_document(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

# 🧠 Vektor-Speicher mit FAISS aufbauen
def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    return FAISS.from_documents(chunks, embeddings)

# 🧠 Nur LLM & Retriever laden
def load_llm_and_retriever(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=google_api_key
    )
    retriever = vectorstore.as_retriever()
    return llm, retriever
