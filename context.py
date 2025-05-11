import os
import sys
import subprocess
import importlib.util
import traceback
import json
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langsmith import Client

# --- Automatische Installation fehlender Pakete ---
required_packages = [
    ("langchain", "langchain"),
    ("langchain_google_genai", "langchain-google-genai"),
    ("langchain_community", "langchain-community"),
    ("chromadb", "chromadb"),
    ("dotenv", "python-dotenv"),
    ("langsmith", "langsmith")
]

for module_name, pip_name in required_packages:
    if not importlib.util.find_spec(module_name):
        print(f"⚠️ Paket '{pip_name}' wird installiert ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"Paket '{pip_name}' erfolgreich installiert.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Fehler bei der Installation von '{pip_name}': {e}")

# --- .env laden und prüfen ---
if not load_dotenv():
    raise FileNotFoundError(".env-Datei fehlt. Bitte erstelle eine .env-Datei mit deinem GOOGLE_API_KEY.")

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY ist nicht gesetzt. Bitte ergänze .env mit z. B. GOOGLE_API_KEY=dein-key")
print(f"GOOGLE_API_KEY geladen: {google_api_key[:5]}...")

# --- LangSmith konfigurieren ---
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")
if langchain_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "ePA_2025_Project"
    os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    print("LangSmith aktiviert.")
else:
    print("LANGCHAIN_API_KEY nicht gefunden – LangSmith Tracing deaktiviert.")

# --- Initialisiere Embedding-Modell ---
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Chat-Historie für Dialogspeicher ---
SESSION_FILE = "session_memory.json"

class FileChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, file_path):
        self.file_path = file_path
        self._messages = self._load_messages()

    def _load_messages(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return [
                    HumanMessage(content=m["content"]) if m["role"] == "human"
                    else AIMessage(content=m["content"])
                    for m in data
                ]
        return []

    def add_message(self, message):
        self._messages.append(message)
        self._save_messages()

    def _save_messages(self):
        data = [
            {"role": "human" if isinstance(m, HumanMessage) else "ai", "content": m.content}
            for m in self._messages[-10:]  # Letzte 10 Nachrichten
        ]
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def clear(self):
        self._messages = []
        if os.path.exists(self.file_path):
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2, ensure_ascii=False)

    @property
    def messages(self):
        return self._messages

def get_memory():
    return FileChatMessageHistory(SESSION_FILE)

def load_and_split_document(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Datei {file_path} wurde nicht gefunden.")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    print(f"{len(chunks)} Textstücke erzeugt.")
    return chunks

def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "epa_2025.txt"
    print(f"📄 Lade Datei: {file_path}")
    try:
        chunks = load_and_split_document(file_path)
        documents = [Document(page_content=chunk, metadata={"chunk_id": f"chunk_{i}"}) for i, chunk in enumerate(chunks)]
        
        if len(chunks) < 50:
            raise ValueError(f"Zu wenig Textblöcke ({len(chunks)}). Erwartet mindestens 50.")
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="epa_2025_collection",
            persist_directory="./chroma_db"
        )
        print("✅ Embeddings erfolgreich erstellt und gespeichert.")
        
        # Speichere initialen Kontext in der Historie
        memory = get_memory()
        memory.add_message(HumanMessage(content="Dokument geladen"))
        memory.add_message(AIMessage(content=f"{file_path} mit {len(chunks)} Chunks verarbeitet."))

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Fehler beim Verarbeiten des Dokuments: {e}")

if __name__ == "__main__":
    main()