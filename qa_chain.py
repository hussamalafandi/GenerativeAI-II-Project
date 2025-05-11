# qa_chain.py — Interactive Chat with Document Retrieval via ChromaDB + Gemini
# Implements Retrieval-Augmented Generation (RAG) with conversational memory using LangChain

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

# 🔐 Load environment variables (e.g., API keys) from .env file
load_dotenv()

# 🧠 Load ChromaDB vector store with precomputed embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever()

# 💬 Initialize memory to track conversation history
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# 📜 Prompt template that includes history and context
prompt = PromptTemplate.from_template("""
Use the following context and conversation history to answer the question.
If the answer is not in the context, say you don’t know.

Chat History:
{chat_history}

Context:
{context}

Question: {question}
Answer:
""")

# 🤖 Load the Gemini model (Flash version)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# 🔗 Define the chain: retrieves context, formats prompt, generates answer
chain = (
    RunnableLambda(lambda x: {
        "context": retriever.get_relevant_documents(x["question"]),
        "question": x["question"],
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })
    | (lambda x: prompt.format(**x))
    | llm
)

# 💬 Interactive command-line interface
print("🔎 Enter your question (or type 'exit' to quit):")
while True:
    question = input("\n🧠 Your question: ")
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # 📊 Enable tracing for LangSmith
    tracer = LangChainTracer()

    # Execute the chain with tracing enabled
    result = chain.invoke(
        {"question": question},
        config={"callbacks": [tracer]}
    )
    print("📄 Answer:", result.content)

    # 🧠 Update memory manually after each turn
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(result.content)
