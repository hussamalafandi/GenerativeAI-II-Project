# RAG-System: 2025 Aktienmarkt-Crash

Ein Retrieval-Augmented Generation (RAG) System, das Informationen über den Aktienmarkt-Crash von 2025 aus einem Wikipedia-Artikel abruft und kontextbasierte Antworten mit einem Sprachmodell generiert.

## ✅ Ziel

- Extrahierung von Inhalten aus einem Wikipedia-Artikel über den Aktienmarkt-Crash von 2025.
- Aufteilung des Inhalts in kleinere Abschnitte und Speicherung dieser in einer persistenten Vektordatenbank (ChromaDB).
- Implementierung eines Dialogflusses zur Beantwortung vordefinierter Fragen basierend auf dem abgerufenen Kontext.
- Verwendung des Sprachmodells `gemini-2.0-flash` zur Generierung von Antworten.

## 🛠️ Technologien

- **LangChain** – Retrieval- und QA-Pipeline.
- **ChromaDB** – Persistente Dokumentenspeicherung.
- **Hugging Face Embeddings** – Text-Embeddings mit `sentence-transformers/all-MiniLM-L6-v2`.
- **LangSmith** – Logging und Tracing.
- **Langraph** – Dialogflussverwaltung.
- **Google GenAI** – Sprachmodell.
- **BeautifulSoup** – Web-Scraping.

## ⚡ Umgebungsvariablen

Setzen Sie die folgenden Umgebungsvariablen:

- `LANGSMITH_API_KEY`
- `TAVILY_API_KEY`
- `GOOGLE_API_KEY`
- `HUGGINGFACE_API_KEY`

Sie können diese Variablen entweder direkt in der Umgebung setzen oder mithilfe der `userdata.get()` Methode in Google Colab.

## 📄 Datenextraktion und Verarbeitung

1. **Web-Scraping**  
   Der Inhalt wird aus dem Wikipedia-Artikel über den Aktienmarkt-Crash von 2025 mit **BeautifulSoup** extrahiert.

2. **Textaufteilung**  
   Der extrahierte Text wird in Abschnitte von 500 Zeichen mit einer Überlappung von 100 Zeichen unterteilt, mithilfe von **LangChain**.

3. **Embedding**  
   Die Abschnitte werden mit **Hugging Face Embeddings** eingebettet, um den Text für die Suche zu repräsentieren.

## 🧠 Dialogfluss

Das System verwendet **Langraph** zur Verwaltung des Dialogflusses, der zwei Hauptschritte umfasst:

1. **Abruf**  
   Abfragen von relevanten Abschnitten basierend auf der Nutzeranfrage.

2. **Generierung**  
   Generierung einer Antwort auf Basis des abgerufenen Kontexts unter Verwendung des Sprachmodells.

## 💬 Fragenbeantwortung

Das System beantwortet vordefinierte Fragen zum Aktienmarkt-Crash von 2025. Für jede Frage wird der Kontext abgerufen und eine Antwort generiert.

## 📝 Protokollierung

Die Interaktionen mit dem System werden mit **LangSmith** protokolliert, um das Tracking und Monitoring zu ermöglichen. Jede Interaktion wird mit Metadaten zur besseren Nachverfolgbarkeit aufgezeichnet.

## 📊 Testen

- Das System wurde mit mindestens fünf sinnvollen Fragen getestet, um die Wirksamkeit der Dokumentabrufstrategie zu überprüfen.
- Jede Frage wird validiert, indem der abgerufene Kontext und die generierte Antwort überprüft werden.

## 🚀 Zukünftige Arbeiten

- Implementierung von Metadatenfilterung zur Verbesserung des Abrufs.
- Ermöglichung der Mehrfachabfrageverarbeitung zur effektiveren Bearbeitung komplexer Fragen.
- Verbesserung der Fehlerbehandlung und Implementierung eines ausgeklügelten Loggings und Monitorings.

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert.
