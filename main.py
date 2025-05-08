from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# 1. Загружаем векторную БД
vectorstore = Chroma(persist_directory="chroma_store", embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 2. Создаём цепочку RAG без агентов
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=retriever,
    return_source_documents=True
)

# 3. Запрос
query = "Что такое квантовая запутанность?"
result = qa_chain.run(query)

print(result)
