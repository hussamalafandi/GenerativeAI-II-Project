# qa_chain.py — интерактивный чат с базой через ChromaDB + Gemini (RAG с памятью через ConversationBufferMemory и интерактивным диалогом)

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langsmith import traceable
from langchain_core.tracers import LangChainTracer

# 🔐 Загрузка переменных из .env
load_dotenv()

# 🧠 Загружаем ChromaDB с эмбеддингами
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever()

# 🧠 Инициализация памяти
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# 📜 Шаблон генерации с контекстом и историей
prompt = PromptTemplate.from_template("""
Используй следующий контекст и историю диалога, чтобы ответить на вопрос.
Если ответа нет в контексте — честно скажи, что не знаешь.

История:
{chat_history}

Контекст:
{context}

Вопрос: {question}
Ответ:
""")

# 🤖 LLM Gemini Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# 🔗 Цепочка с использованием памяти
chain = (
    RunnableLambda(lambda x: {
        "context": retriever.get_relevant_documents(x["question"]),
        "question": x["question"],
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })
    | (lambda x: prompt.format(**x))
    | llm
)

# 💬 Интерактивный режим
print("🔎 Введите вопрос (или 'выход' для завершения):")
while True:
    question = input("\n🧠 Ваш вопрос: ")
    if question.lower() in ["выход", "exit", "quit"]:
        print("👋 До встречи!")
        break
    
    # Создаём трассировщик
    tracer = LangChainTracer()
    
    # Запуск с трассировкой
    result = chain.invoke(
    {"question": question},
    config={"callbacks": [tracer]})
    print("📄 Ответ:", result.content)

    # Добавляем в память вручную (симулируем сохранение истории)
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(result.content)