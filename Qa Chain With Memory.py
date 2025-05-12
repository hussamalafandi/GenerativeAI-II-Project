# qa_chain_with_memory.py — Interactive RAG Chat with Memory + LangSmith Tracing

import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.tracers import LangChainTracer

# 🔐 Load environment variables
load_dotenv()

# 🧠 Load ChromaDB with embeddings
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever()

# 🧠 Initialize chat memory
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history"
)

# 📜 Prompt with context and chat history
prompt = PromptTemplate.from_template("""
Use the following context and chat history to answer the question.
If the answer is not in the context, just say you don't know.

History:
{chat_history}

Context:
{context}

Question: {question}
Answer:
""")

# 🤖 Load Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# 🔗 Build RAG chain with memory
rag_chain = (
    RunnableLambda(lambda x: {
        "context": retriever.get_relevant_documents(x["question"]),
        "question": x["question"],
        "chat_history": memory.load_memory_variables({})["chat_history"]
    })
    | (lambda x: prompt.format(**x))
    | llm
)

# 🧪 Enable LangSmith tracing safely
tracer = LangChainTracer(project_name="RAG-Chat-With-Memory")

# 💬 Interactive Q&A loop
print("Ask your question (type 'exit' to quit):")
while True:
    question = input("\n❓ Question: ")
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    result = rag_chain.invoke(
        {"question": question},
        config={"callbacks": [tracer]}
    )

    print("📄 Answer:", result.content)

    # Store in memory manually
    memory.chat_memory.add_user_message(question)
    memory.chat_memory.add_ai_message(result.content)

