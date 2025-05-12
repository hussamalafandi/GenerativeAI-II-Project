# =============================
# 📄 Document Indexing Script
# =============================

import os
import shutil
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# -----------------------------
# 1. Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# 2. Clean up previous index
# -----------------------------
if os.path.exists("chroma_store"):
    shutil.rmtree("chroma_store")
    print("🧹 Removed old ChromaDB index.")

# -----------------------------
# 3. Load document (news article)
# -----------------------------
with open("./data/biocomputer_article.txt", "r", encoding="cp1252") as file:
    raw_text = file.read()

# -----------------------------
# 4. Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_text(raw_text)

# Wrap with metadata
documents = [Document(page_content=t, metadata={"source": "biocomputer_article"}) for t in texts]

print(f"✅ Split into {len(documents)} chunks.")

# -----------------------------
# 5. Create embedding model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------
# 6. Store in ChromaDB with persistence
# -----------------------------
vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=embedding_model
)
vectorstore.add_documents(documents)

print("✅ Indexed and persisted successfully.")
