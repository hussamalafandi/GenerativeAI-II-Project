# 🔍 Generative QA mit LangChain, Gemini 2 & Web-Retrieval

Dieses Projekt ist ein Frage-Antwort-System basierend auf **Google Gemini 2.0**, **LangChain**, **LangSmith Tracing** und einer Sammlung aktueller Web-Artikel. Es kombiniert direkte LLM-Antworten mit dokumentenbasiertem Retrieval und speichert den Gesprächskontext.

---

## 🚀 Funktionen

- Verwendung von **Google Gemini 2.0 Flash** (LLM)
- Beantwortung aktueller Fragen mit und ohne Retrieval
- Integration von Webartikeln aus Nachrichtenquellen (DW, Tagesschau, Wikipedia, ZDF usw.)
- Speicherung von Fragen und Antworten im lokalen JSON-Verlauf
- Kontextbasierte Antwortgenerierung über LangChain + LangSmith
- Embedding & Vektor-Suche via **ChromaDB**

---

## 📦 Voraussetzungen

- Python 3.10+ (Colab empfohlen)
- API-Zugänge:
  - [Google AI / Gemini API](https://ai.google.dev/)
  - [LangSmith](https://smith.langchain.com/o/77078b25-f602-4022-9037-17871b75fbb5/projects/p/fe0e8900-acc2-4dda-a081-38bc677cf9cf?timeModel=%7B%22duration%22%3A%227d%22%7D)
- Installiere benötigte Pakete:

```bash
pip install langchain langsmith langchain_google_genai langchain_community beautifulsoup4 chromadb
