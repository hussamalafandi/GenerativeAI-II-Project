# %% [markdown]
# 
# #  RAG-System mit LangChain, ChromaDB und Gemini 2
# 
from dotenv import load_dotenv
import os

load_dotenv()  # <-- подхватит переменные из .env

HF_TOKEN = os.environ["HF_TOKEN"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]




# %%
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI


import os

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)




# %% [markdown]
# # Seite laden

# %%
url = "https://en.wikipedia.org/wiki/Reopening_of_Notre-Dame_de_Paris?utm_source=chatgpt.com"
loader = WebBaseLoader(url)
documents = loader.load()

# %% [markdown]
# # Text in Chunks aufteilen

# %%
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=120
)
chunks = splitter.split_documents(documents)
print(f"Anzahl der Chunks: {len(chunks)}")

# %% [markdown]
# # Vektorisierung und Speicherung in ChromaDB

# %% [markdown]
# embedding

# %%
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# %% [markdown]
# chromaDB

# %%


persist_directory = "chromadb"
if os.path.exists(persist_directory) and os.listdir(persist_directory):
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)
else:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectorstore.persist()


# %% [markdown]
# langsmith

# %%
os.environ["LANGCHAIN_PROJECT"] = "Reopening of Notre-Dame de Paris"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
from langchain.callbacks.tracers import LangChainTracer

tracer = LangChainTracer(project_name="Reopening of Notre-Dame de Paris")


# %% [markdown]
# class State erstellen

# %%
from langchain_core.documents import Document
from typing_extensions import List, TypedDict


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# %% [markdown]
# 
# #  Dialogsystem und Memory

# %%
from langchain.memory.chat_message_histories import FileChatMessageHistory
message_history = FileChatMessageHistory("chathistory.json")

memory = ConversationBufferMemory(
    memory_key="chathistory",
    chat_memory=message_history,
    return_messages=True
)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")



# %% [markdown]
# Prompt erstellen

# %%
from langchain import hub

prompt = hub.pull("rlm/rag-prompt")

#prompt = hub.pull("langchain-ai/retrieval-qa-chat")





# %% [markdown]
# Finktionen retrieve und generate

# %%

def retrieve(state: State):
    retrieved_docs = vectorstore.similarity_search(state["question"])
    return {"context": retrieved_docs}




def generate(state: State):
    #  Lade bisherigen Gesprächsverlauf aus der Datei
    history = memory.load_memory_variables({})["chathistory"]
    

    #  Bereite die Dokumente für den Prompt vor
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    print(state['context'])

    #  Kombiniere Kontext + Chatverlauf im Prompt
    full_context = f"{history}\n\nDokumente:\n{docs_content}"
    
    #  Erzeuge Prompt mit LangChain Hub-Vorlage
    
    messages = prompt.invoke({
    "question": state["question"],
    "context": full_context
    })

    #  LLM generiert eine Antwort
    response = llm.invoke(messages, config={"callbacks": [tracer]})

    #  Speichere aktuelle Frage und Antwort im Speicher (für nächste Runde)
    memory.save_context(
        inputs={"input": state["question"]},
        outputs={"output": response.content}
    )

    #  Rückgabe für das nächste Graph-State
    return {"answer": response.content}

# %% [markdown]
# Graph

# %%
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# %%
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

# %% [markdown]
# Test

# %% [markdown]
# # Beispiel-Dialog

# %%

fragen = [
    "When was Notre Dame Cathedral reopened?",
    "Who led this ceremony?",
    "Who was present at this ceremony?",
    "what he said in his speech",
    "did he also participate in the celebrations the next day?"
     
]



# %%
for frage in fragen:
    
    result = graph.invoke({"question": frage})
    print(f'Frage: {frage}')
    print(f'Context: {result["context"]}')
    print(f'Antwort: {result["answer"]}\n\n')


