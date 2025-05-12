# RAG System: 2025 Stock Market Crash

A Retrieval-Augmented Generation (RAG) system that retrieves information about the 2025 stock market crash from a Wikipedia article and generates context-based answers using a language model.

## ✅ Goal

- Extracting content from a Wikipedia article about the 2025 stock market crash.
- Splitting the content into smaller chunks and storing them in a persistent vector database (ChromaDB).
- Implementing a dialogue flow for answering predefined questions based on the retrieved context.
- Using the `gemini-2.0-flash` language model to generate responses.

## 🛠️ Technologies

- **LangChain** – Retrieval and QA pipeline.
- **ChromaDB** – Persistent document storage.
- **Hugging Face Embeddings** – Text embeddings with `sentence-transformers/all-MiniLM-L6-v2`.
- **LangSmith** – Logging and tracing.
- **Langraph** – Dialogue flow management.
- **Google GenAI** – Language model.
- **BeautifulSoup** – Web scraping.

## ⚡ Environment Variables

Set the following environment variables:

- `LANGSMITH_API_KEY`
- `TAVILY_API_KEY`
- `GOOGLE_API_KEY`
- `HUGGINGFACE_API_KEY`

You can set these variables either directly in the environment or using the `userdata.get()` method in Google Colab.

## 📄 Data Extraction and Processing

1. **Web Scraping**  
   The content is scraped from the Wikipedia article about the 2025 stock market crash using **BeautifulSoup**.

2. **Text Splitting**  
   The extracted text is split into chunks of 500 characters with an overlap of 100 characters, using **LangChain**.

3. **Embedding**  
   The chunks are embedded using **Hugging Face Embeddings** to represent the text for search.

## 🧠 Dialogue Flow

The system uses **Langraph** to manage the dialogue flow, which consists of two main steps:

1. **Retrieval**  
   Querying relevant sections based on the user's query.

2. **Generation**  
   Generating an answer based on the retrieved context using the language model.

## 💬 Question Answering

The system answers predefined questions about the 2025 stock market crash. For each question, the context is retrieved, and an answer is generated.

## 📝 Logging

Interactions with the system are logged using **LangSmith** for tracking and monitoring. Each interaction is recorded with metadata for better traceability.

## 📊 Testing

- The system has been tested with at least five meaningful questions to validate the effectiveness of the document retrieval strategy.
- Each question is validated by checking the retrieved context and the generated answer.

## 🚀 Future Work

- Implementing metadata filtering to improve retrieval.
- Enabling multi-query processing for more effective handling of complex questions.
- Enhancing error handling and implementing advanced logging and monitoring.

## 📄 License

This project is licensed under the MIT License.
