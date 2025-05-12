
# Homework Project: Build a RAG (Retrieval-Augmented Generation) System

## Objective
Develop a basic Retrieval-Augmented Generation (RAG) system that retrieves information from an external document source and uses it to answer questions. This will demonstrate how language models can be grounded in up-to-date, domain-specific knowledge.

> The chosen model has a knowledge cutoff in **August 2024**, so answers to recent topics must rely on **retrieved documents**, not internal model knowledge.

---

## Core Requirements

### 1. Document Indexing
- Use **Chromadb** with **persistence enabled**.
- Choose a document of an event that happend after **August 2024**
- Include document **splitting** (≥ 50 chunks) using appropriate text splitting strategies.

### 2. System Components
- Use `gemini-2.0-flash`
- Implement with **LangChain** or **LlamaIndex**.  
- Use **LangSmith** or **LangFuse**.  
- Use **Git and GitHub** for version control.
- Don not use pre-built agents.
- **Dialog flow** (multi-turn interaction)  
- **Memory** (context tracking across interactions)  

### 3. Experimentation  
- Compare **system prompts** and their effects on model behavior.  
- Use a variety of **questions** to evaluate system robustness (at leat 5 different questions with correct answers).

### 4. Reproducibility  
- Submit your code via **GitHub**.  
- Use a clean repository:  
  - ❌ **No large files** in git history  
  - ❌ **No secret tokens** in commit history

### 5. Submission  
- **Deadline:** `11.05 at 23:59`  
- **Deliverables:**
  - GitHub repo link
  - Link to your **LangSmith** or **LangFuse** project.
  - Jupyter notebook or script demonstrating:
    - Index creation
    - Retrieval
    - Answer generation
    - Prompt variations

---

## Bonus Features (Optional, for Extra Credit)

Implement one or more of the following to enhance your RAG system:

- ✅ **Metadata filtering** during document retrieval  
- ✅ **Multi-Query retrieval** (ask multiple questions or rephrase to get better context)

---
=======
# 🤖 GenerativeAI-II-Projekt: RAG mit Gemini 2 & LangChain

## 📝 Projektbeschreibung

Dieses Projekt implementiert ein **RAG-System (Retrieval-Augmented Generation)** mit **Gemini 2 (Google)** und der **LangChain**-Bibliothek. Es lädt Artikel aus dem Web, extrahiert relevante Informationen, vektorisiert die Inhalte und ermöglicht dialogbasierte Antworten auf Deutsch.

---

## 🔧 Installationsanleitung

### 1. ✅ Abhängigkeiten installieren

```bash
pip install -q langchain langchain-community langchain-google-genai langgraph chromadb

2. ✅ Umgebungsvariablen setzen
import os
from getpass import getpass

os.environ["LANGCHAIN_API_KEY"] = getpass("🔑 LangSmith API Key eingeben: ")
os.environ["LANGCHAIN_PROJECT"] = "RAG.2025"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["GOOGLE_API_KEY"] = getpass("🔑 Google API Key (Gemini) eingeben: ")
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/123.0.0.0 Safari/537.36"

🔍 Datenquelle
Das System lädt Inhalte von dieser URL (Webartikel):
https://www.astromind.de/astrologie-artikel/r%C3%BCcklaeufiger-mars.html


🧠 Was passiert im Code?
🔹 Laden & Aufteilen: Webseite wird geladen, in kleine Texte (Chunks) zerlegt.

🔹 Vektorisierung: Jeder Chunk wird mittels HuggingFace-Embedding in einen Vektor umgewandelt.

🔹 Speicherung: Die Vektoren werden lokal in einer chroma_db gespeichert.

🔹 Speicher: Gesprächsverlauf wird in chat_history.json gespeichert.

🔹 Retrieval: Mehrere Suchanfragen pro Frage werden erzeugt.

🔹 Antwortgenerierung: Die Antwort basiert auf Dokumentinhalten und Gesprächskontext.

🤖 Modell & Tools
LLM: Gemini 2.0 Flash von Google

Retriever: MultiQueryRetriever aus LangChain

Prompt: Eingebunden von LangChain Hub (rlm/rag-prompt)

Tracer: Aktiv für LangSmith zur Nachverfolgung.

🚀 Ausführung (Beispiel)
state = {"question": "frage ?"}
result = graph.invoke(state)
print("📍Antwort:", result["answer"])


👤 Autorin
Alona Tkachenko
🔗 GitHub: altkachenko11
🔗 **LangSmith Projekt-Tracking:**  
[smith.langchain.com Projekt-Link](https://smith.langchain.com/o/75759cbe-275f-4bca-8856-b15c344abcf9/projects/p/47040df8-9915-40ae-90be-a6c0a66d0b52?timeModel=%7B%22duration%22%3A%227d%22%7D)


