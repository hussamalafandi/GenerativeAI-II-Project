# qa_chain.py — RAG with Gemini, ChromaDB, and Chat History

import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Load environment variables
load_dotenv()

# Load ChromaDB with embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever()

# Initialize conversation memory
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# Prompt template with history
prompt = PromptTemplate.from_template("""
You are an expert assistant. Use the following context and chat history to answer the user's question.
If the answer is not found in the context, reply with "I don't know."

Chat history:
{chat_history}

Context:
{context}

Question: {question}
Answer:
""")

# LLM (Gemini Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# RAG chain with memory
rag_chain_with_memory = (
    RunnableLambda(lambda x: {
        "context": retriever.get_relevant_documents(x["question"]),
        "question": x["question"],
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })
    | (lambda x: prompt.format(**x))
    | llm
)

# Interactive chat
print("\n🔎 Ask your question (type 'exit' to quit):")
while True:
    question = input("\n❓ Question: ")
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    result = rag_chain_with_memory.invoke({"question": question})
    print("📄 Answer:", result.content)

    # Store exchange in memory
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(result.content)

