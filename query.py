import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

# Load and split document
def load_and_split_document(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk, metadata={"chunk_id": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]
    return documents

# Build vector store with Chroma
def build_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=google_api_key
    )
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="epa_2025_collection",
        persist_directory="./chroma_db"
    )

# Load LLM and retriever
def load_llm_and_retriever(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=google_api_key
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return llm, retriever