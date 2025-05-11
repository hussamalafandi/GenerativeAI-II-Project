This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using ChromaDB, Gemini (Google Generative AI), and LangChain with conversational memory.

## 📚 Project Objective

The goal is to answer user questions based on a real-world document about events after **August 2024** using retrieval-based context. The model alone doesn't have access to this knowledge and must rely on indexed documents.

---

## ✅ Core Features

- 🔍 **Document Indexing** with ChromaDB and HuggingFace Embeddings
- 🤖 **LLM** using `gemini-2.0-flash` from Google Generative AI
- 🔗 **LangChain Pipeline** (no agents used)
- 💬 **Multi-turn dialogue** with conversational memory
- 📦 **Persistence** of vector store
- 🧪 **Traceability** via LangSmith

---

## 📁 File Structure

```
GenerativeAI-II-Project/
├── .env
├── requirements.txt
├── indexing.py         # Indexes document chunks to ChromaDB
├── qa_chain.py         # Main interactive chatbot script with memory
├── chroma_store/       # Folder where ChromaDB persists data
└── biocomputer_article.txt  # Source document
```

---

## ⚙️ Setup

### 1. Clone and create virtual environment:

```bash
git clone https://github.com/yourusername/GenerativeAI-II-Project.git
cd GenerativeAI-II-Project
python -m venv rag_env_310
.
ag_env_310\Scriptsctivate
pip install -r requirements.txt
```

### 2. Set up `.env` with your API keys:

```ini
GOOGLE_API_KEY=your_google_api_key
LANGSMITH_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=RAG-Chat-Memory
```

---

## 🚀 How to Run

### Step 1: Index the document

```bash
python indexing.py
```

This will split your document into chunks and store them with embeddings in ChromaDB.

### Step 2: Start the chatbot

```bash
python qa_chain.py
```

Then simply enter your questions in the terminal.

---

## 🧠 Example Questions

- Who created the biocomputer?
- When was it first introduced?
- What does this mean for the future of AI?

---

## 📈 Bonus Features

✅ Conversational memory  
✅ LangSmith tracing  
❌ (Optional features like metadata filtering and multi-query not included in final version)

---

## 🔗 LangSmith Trace Example

Visit: [https://smith.langchain.com](https://smith.langchain.com) and find the project `RAG-Chat-Memory`.

---

## 🧼 Clean Code Checklist

- [x] No API keys committed
- [x] `.env` listed in `.gitignore`
- [x] Code modular and documented
- [x] No large files in Git history

---

## 📅 Deadline

Submission: **11.05, 23:59**

Good luck! 🚀
