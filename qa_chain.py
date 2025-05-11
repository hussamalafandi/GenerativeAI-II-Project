# ===============================================================
# 📁 Step 1: Load environment variables / Umgebungsvariablen laden
# ===============================================================
from dotenv import load_dotenv
import os
load_dotenv()

# ===============================================================
# 🤖 Step 2: Load the LLM model / LLM-Modell laden
# ===============================================================
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# ===============================================================
# 🔤 Step 3: Load embeddings / Embeddings laden
# ===============================================================
from langchain_huggingface import HuggingFaceEmbeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ===============================================================
# 💾 Step 4: Load document index with Chroma and apply metadata filter / Index mit Metadatenfilter laden
# ===============================================================
from langchain_chroma import Chroma
vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=embedding_model
)

# ===============================================================
# 🔁 Step 5: Set up retriever with metadata filtering and multi-query / Retriever mit Filter und Multi-Query erstellen
# ===============================================================
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": {"source": "biocomputer_article"}}
    ),
    llm=llm
)

# ===============================================================
# ✏️ Step 6: Prompt Template / Prompt-Vorlage erstellen
# ===============================================================
from langchain.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template(
    """
    Answer the question using ONLY the following context:
    Beantworte die Frage NUR anhand des folgenden Kontexts:

    {context}

    Question / Frage: {question}
    """
)

# ===============================================================
# 🧠 Step 7: Setup conversation memory / Dialogspeicher einrichten
# ===============================================================
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
history = FileChatMessageHistory(file_path="chat_history.json")
memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=history, return_messages=True)

# ===============================================================
# 🔗 Step 8: Build the RAG chain / RAG-Kette bauen
# ===============================================================
from langchain.chains import RunnableWithMessageHistory
chain = (
    {"context": lambda x: retriever.invoke(x["question"]), "question": lambda x: x["question"]}
    | prompt
    | llm
)

chat_with_memory = RunnableWithMessageHistory(
    chain,
    lambda session_id: memory,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# ===============================================================
# 📊 Step 9: Tracing with LangSmith / Nachverfolgung aktivieren
# ===============================================================
from langchain_core.tracers import LangChainTracer
tracer = LangChainTracer()

# ===============================================================
# 💬 Step 10: Interactive Q&A Loop / Interaktives Frage-Antwort-Menü
# ===============================================================
print("\U0001F50D Enter your question (or type 'exit' to quit):")
print("\U0001F50D Gib deine Frage ein (oder tippe 'exit' zum Beenden):")

while True:
    question = input("\n\U0001F9E0 Your question / Deine Frage: ")
    if question.lower() in ["exit", "quit"]:
        print("\U0001F44B Goodbye! / Auf Wiedersehen!")
        break

    result = chat_with_memory.invoke(
        {"question": question},
        config={"configurable": {"session_id": "user"}, "callbacks": [tracer]}
    )

    print("\U0001F4C4 Answer / Antwort:", result.content)
