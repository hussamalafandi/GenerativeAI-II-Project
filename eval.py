import json
import os
from dotenv import load_dotenv
from query import load_and_split_document, build_vector_store, load_llm_and_retriever
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import MultiQueryRetriever

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "ePA_2025_Project"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") or ""

file_path = "epa_2025.txt"

# Load and prepare document
chunks = load_and_split_document(file_path)
vectorstore = build_vector_store(chunks)
llm, base_retriever = load_llm_and_retriever(vectorstore)

# Configure MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm
)

# Define prompt
prompt = PromptTemplate.from_template("""
Du bist ein hilfreicher Assistent. Beantworte die Frage basierend auf dem gegebenen Kontext.

Kontext:
{context}

Frage:
{question}
""")

# Test questions
testfragen = [
    "Was steht im ePA-Entwurf 2025 zur Nutzung von Gesundheitsdaten?",
    "Wie wird die Rolle der Gematik im Jahr 2025 beschrieben?",
    "Welche neuen Zugriffsrechte werden im ePA-Update 2025 geregelt?",
    "Was plant die Bundesregierung im ePA-Bereich für die Forschung ab 2025?",
    "Welche Fristen gelten für Ärztinnen und Ärzte laut ePA-2025-Gesetz?"
]

# Results
results = []

def rag_antwort(question):
    docs = retriever.invoke(question)
    print(f"\n🔍 Retrieved Documents for '{question}':")
    for i, doc in enumerate(docs, 1):
        print(f"Doc {i}: {doc.page_content[:200]}... (Metadata: {doc.metadata})")
    context = "\n\n".join([doc.page_content for doc in docs])
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content.strip()

def baseline_antwort(question):
    chain = prompt | llm
    response = chain.invoke({"context": "", "question": question})
    return response.content.strip()

# Evaluation
for frage in testfragen:
    print(f"\n🔍 Frage: {frage}")
    rag = rag_antwort(frage)
    base = baseline_antwort(frage)
    print("📚 Mit Retrieval:", rag)
    print("🧠 Ohne Retrieval:", base)
    results.append({
        "frage": frage,
        "antwort_mit_retrieval": rag,
        "antwort_ohne_retrieval": base
    })

# Save results
with open("eval_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\n✅ Evaluation abgeschlossen. Ergebnisse in eval_results.json gespeichert.")