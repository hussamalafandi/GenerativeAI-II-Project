import os
import sys
import subprocess
import importlib.util

# --- Automatische Installation fehlender Pakete ---
required_packages = [
    ("langchain", "langchain"),
    ("langchain_google_genai", "langchain_google_genai"),
    ("langchain_community", "langchain_community"),
    ("chroma", "langchain-chroma"),
    ("dotenv", "python-dotenv"),
    ("langsmith", "langsmith")  # optional, wenn du LangSmith verwenden möchtest
]

for module_name, pip_name in required_packages:
    if not importlib.util.find_spec(module_name):
        print(f"⚠️ Paket '{pip_name}' wird installiert ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])

# --- Imports nach Installation ---
import traceback
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema import Document  # Importiere das Document-Objekt von LangChain
from langchain.memory import ConversationBufferMemory  # Für die Dialogführung

# Initialisiere das Embedding-Modell
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def load_and_split_document(file_path):
    # Lade die Datei und teile sie in Textblöcke auf
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)

    return chunks

def main():
    # Textdatei optional per Argument oder Standardwert
    file_path = sys.argv[1] if len(sys.argv) > 1 else "epa_2025.txt"
    print(f"📄 Lade Datei: {file_path}")
    try:
        chunks = load_and_split_document(file_path)
        
        # Füge chunk_id zu jedem Chunk hinzu und erstelle Document-Objekte
        documents = [Document(page_content=chunk, metadata={"chunk_id": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]
        
        if len(chunks) < 50:
            raise ValueError(f"Zu wenig Textblöcke ({len(chunks)}). Bitte überprüfe den Inhalt der Datei.")
        
        # Vektor-Datenbank erstellen und speichern
        vector_store = Chroma.from_documents(
            documents=documents,  # Verwende die Dokument-Objekte
            embedding=embeddings,
            collection_name="epa_2025_collection",
            persist_directory="./chroma_db"
        )
        print("✅ Embeddings erfolgreich erstellt und gespeichert.")
    
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Fehler beim Verarbeiten des Dokuments: {e}")

if __name__ == "__main__":
    main()
