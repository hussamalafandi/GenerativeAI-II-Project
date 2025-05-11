# qa_chain.py — интерактивный чат с базой через ChromaDB + Gemini

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

# 🔐 Загрузка переменных из .env
load_dotenv()

# 🧠 Загружаем ChromaDB с эмбеддингами
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever()

# 📜 Шаблон генерации с контекстом
prompt = PromptTemplate.from_template("""
Используй следующий контекст, чтобы ответить на вопрос.
Если ответ не содержится в контексте — скажи честно, что не знаешь.

Контекст:
---------
{context}
---------

Вопрос: {question}
Ответ:
""")

# 🤖 LLM Gemini Flash
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# 🔗 Цепочка: получение контекста → шаблон → модель
chain = (
    RunnableLambda(lambda x: {
        "context": retriever.get_relevant_documents(x["question"]),
        "question": x["question"]
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
    result = chain.invoke({"question": question})
    print("📄 Ответ:", result.content)
