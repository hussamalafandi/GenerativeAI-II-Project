from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from query import load_and_split_document, build_vector_store, load_llm_and_retriever
from langsmith import Client
import os
from dotenv import load_dotenv
try:
    from context import FileChatMessageHistory
except ImportError as e:
    print(f"Fehler beim Import von FileChatMessageHistory: {e}")
    raise

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ePA_2025_Project"

file_path = "epa_2025.txt"
SESSION_FILE = "session_memory.json"

# Load vector store and LLM
chunks = load_and_split_document(file_path)
vectorstore = build_vector_store(chunks)
llm, base_retriever = load_llm_and_retriever(vectorstore)

# Configure retriever
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

# Define prompt
prompt = PromptTemplate.from_template("""
Du bist ein Experte für das deutsche Gesundheitswesen. Beantworte die Frage präzise basierend auf dem Kontext und der Gesprächshistorie. Wenn keine relevanten Informationen vorliegen, sage es deutlich.
Kontext:
{context}
Gesprächshistorie:
{chat_history}
Frage:
{question}
""")

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    chat_memory=FileChatMessageHistory(SESSION_FILE),
    output_key="answer"
)

# Create conversational RAG chain
rag_chain_traced = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt},
    return_source_documents=True
)

# Interactive loop
client = Client()
print("ePA-2025 Q&A System. Type 'exit' to quit.")
while True:
    query = input("Frage: ")
    if query.lower() == "exit":
        break
    try:
        result = rag_chain_traced.invoke({"question": query})
        response_text = result['answer']
        print(f"Antwort: {response_text}")

        # Log generated queries
        try:
            generated_queries = retriever.generate_queries(query, llm)
            print(f"\n🔍 Generated Queries: {generated_queries}")
        except AttributeError:
            print("\n🔍 Generated Queries: Not available")

        # Log feedback if incomplete
        if "keine Informationen" in response_text.lower():
            client.create_feedback(
                run_id=result.get("run_id", None),
                key="incomplete_answer",
                score=0.0,
                comment=f"Incomplete answer for query: {query}"
            )

        # Print sources
        print("\nQuellen:")
        for i, doc in enumerate(result['source_documents'], 1):
            chunk_id = doc.metadata.get('chunk_id', 'Unknown')
            print(f"{i}. Chunk {chunk_id}: {doc.page_content[:200]}...")
    except Exception as e:
        print(f"Fehler: {e}. Versuche es erneut oder überprüfe die API-Quota.")