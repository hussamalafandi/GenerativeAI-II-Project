import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers import LangChainTracer

# ========================================
# 🔐 Load API keys from .env
# ========================================
load_dotenv()

# ========================================
# 🧠 Load Chroma vector store
# ========================================
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(
    persist_directory="chroma_store",
    embedding_function=embedding_model
)

# Apply metadata filtering to restrict retrieval to only relevant chunks
filtered_retriever = vectorstore.as_retriever(search_kwargs={
    "filter": {"source": "biocomputer_article"}  # Match the metadata used in indexing
})

# Wrap in MultiQueryRetriever to improve results
retriever = MultiQueryRetriever.from_llm(
    retriever=filtered_retriever,
    llm=ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )
)

# ========================================
# 🧾 Prompt template
# ========================================
prompt = PromptTemplate.from_template("""
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:
""")

# ========================================
# 🤖 Gemini model for answer generation
# ========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3
)

# ========================================
# 🔗 RAG Chain (no memory, multi-query retrieval)
# ========================================
rag_chain = (
    RunnableLambda(lambda x: {
        "context": retriever.get_relevant_documents(x["question"]),
        "question": x["question"]
    })
    | (lambda x: prompt.format(**x))
    | llm
)

# ========================================
# 🚀 Interactive loop with tracing enabled
# ========================================
print("\nAsk your question (type 'exit' to quit):")
while True:
    question = input("\n❓ Question: ")
    if question.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    tracer = LangChainTracer()
    result = rag_chain.invoke({"question": question}, config={"callbacks": [tracer]})

    print("\U0001F4C4 Answer:", result.content)
