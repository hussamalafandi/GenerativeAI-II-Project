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


# 🧠 Reward Hacking RAG-Projekt

Dieses Projekt wurde im Rahmen eines LLM-Seminars entwickelt und demonstriert ein **Retrieval-Augmented Generation (RAG)**-System zur Beantwortung komplexer Fragen rund um das Thema **Reward Hacking** bei KI-Modellen.

Als Datenbasis dient der Blogpost von Lilian Weng:  
👉 [Reward Hacking in Reinforcement Learning](https://lilianweng.github.io/posts/2024-11-28-reward-hacking/)

---

## 🎯 Ziel des Projekts

Ziel ist es, ein interaktives System zu bauen, das:
- Informationen aus einem Fachtext extrahiert
- Nutzeranfragen über ein LLM beantwortet
- Dokumentrecherche, Chunking und Embeddings nutzt
- Den gesamten Antwortprozess über **LangGraph** visualisiert

---

## 🔗 Live-Demo via LangSmith

> 👉 **[Hier klicken, um den Workflow in LangSmith zu sehen](https://smith.langchain.com/o/ee2afb6b-cdb5-4358-b346-3131497a9ff4/projects/p/ba5b2e9f-2cc4-4004-823d-455e1c75713e?timeModel=%7B%22duration%22%3A%227d%22%7D)**  
> Die Konversationen, Tool-Nutzung, Retrieval-Calls und Generierungsschritte sind dort transparent nachvollziehbar.

---

## ⚙️ Verwendete Technologien

- **LangChain** – Toolbindung & Promptstruktur
- **LangGraph** – Konversationsfluss & Memory
- **Chroma** – Vektor-Datenbank
- **HuggingFace Embeddings** – `all-mpnet-base-v2`
- **Google Gemini 2.0 (via `init_chat_model`)**
- **LangSmith** – Logging & Debugging der Pipeline

---

## 💡 Beispiel-Fragen

```text
Frage: Was ist ein Beispiel für Reward Tampering?
Antwort: Der Agent verändert direkt die Belohnungsfunktion, um sich selbst höhere Belohnung zuzuweisen.

