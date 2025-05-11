
# Gemini Project 2025 🌌

An advanced natural language processing (NLP) pipeline for interactive document retrieval, conversation management, and knowledge extraction, using **LangChain**, **Google PaLM (Gemini-2.0-flash)**, and **ChromaDB**. This project leverages multi-turn dialogue management, custom embeddings, and graph-based conversation visualization.

---

## **📝 Project Overview**

This project implements a full pipeline for document-based question answering using **LangChain**, **LangSmith**, and **ChromaDB**. It features:

- Document chunking and indexing with ChromaDB
- Real-time conversation memory
- Multi-query retrieval
- Graph-based conversation flow visualization
- Chat history export in both plain text and JSON formats

---

## **📦 Installation**

Ensure you have the required libraries installed:

```bash
pip install -qU "langchain[google-genai]" chromadb networkx matplotlib
pip install langchain-community
pip install -qU sentence-transformers transformers langchain-huggingface langchain-chroma langchain
```

---

## **🚀 How It Works**

1. **Document Indexing:**
   - Downloads a news article.
   - Splits it into at least 50 chunks using **RecursiveCharacterTextSplitter**.
   - Indexes the chunks using **ChromaDB** with persistent storage.

2. **Conversation Management:**
   - Handles multi-turn interactions using **ConversationBufferMemory**.
   - Tracks user inputs and AI responses for context-aware conversations.

3. **Graph-Based Visualization:**
   - Builds a directed graph of user questions and AI responses.
   - Visualizes the conversation flow for better insight into model behavior.

4. **Chat History Export:**
   - Saves conversation history in **chat_history.txt** (plain text) and **chat_history.json** (structured JSON).

---

## **🛠️ Setup and Configuration**

### **🔑 API Keys**
Make sure you have the following environment variables set:

- **LANGSMITH_API_KEY** (required)
- **GOOGLE_API_KEY** (required)

The script will prompt you to enter these keys if they are not set.

---

## **📝 Code Structure**

### **1. Environment Setup and Imports**
- Imports essential libraries.
- Sets environment variables for LangSmith tracing.

### **2. Document Download and Chunking**
- Downloads the target article.
- Splits it into manageable chunks.

### **3. ChromaDB Initialization**
- Creates a ChromaDB instance for efficient document retrieval.

### **4. Language Model Initialization**
- Initializes the **Gemini-2.0-flash** model for NLP tasks.

### **5. Retrieval Chain Initialization**
- Configures the document retriever for accurate response generation.

### **6. Conversation Management**
- Sets up conversation memory to track multi-turn interactions.

### **7. Graph-Based Flow Visualization**
- Creates a directed graph to visualize conversation flow.

### **8. Chat History Export**
- Saves chat history to both text and JSON formats for analysis.

---

## **📝 Sample Questions**

The code includes sample questions to test the system:

- **What was the main reason for Friedrich Merz's defeat?**
- **How did the German parliament react to the vote?**
- **What impact will this have on German politics?**
- **Were there any unexpected outcomes from this vote?**
- **What were the main political alliances mentioned in the article?**

---

## **📊 Graph Visualization**

The conversation flow is visualized using **NetworkX** and **matplotlib**, providing a clear, graphical representation of user and AI interactions.

---

## **📂 Saving Chat History**

Chat history is automatically saved to:

- **chat_history.txt** (plain text)
- **chat_history.json** (structured JSON)

Both files are created in the current working directory.

---

## **📝 Future Improvements**

- Add support for richer conversation context.
- Integrate confidence scoring for AI responses.
- Implement response ranking for better answer quality.
- Optimize graph layout for clearer conversation flows.

---

## **🧑‍💻 Contributing**

Feel free to fork this repository, submit pull requests, and share your ideas for improvement.

---

## **📝 License**

This project is licensed under the MIT License.

---

## **❤️ Acknowledgements**

Special thanks to the **LangChain**, **ChromaDB**, and **Google PaLM** teams for their powerful tools and libraries.

---

Enjoy building with the power of Gemini! 🚀
