# indexing.py — индексирование документа в ChromaDB с persistence

import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv

# 🔐 Загрузка API-ключей
load_dotenv()

# 📄 Загружаем текст документа (предположим, он лежит в .txt)
with open("./data/brain_article.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# 🔹 Разделение текста на чанки
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ".", " "]
)
chunks = text_splitter.split_text(raw_text)

# 🧱 Преобразуем чанки в объекты Document с метаданными
source_metadata = {
    "source": "newatlas.com",
    "date": "2024-09-18",
    "title": "Cortical Bioengineered Intelligence"
}
documents = [Document(page_content=chunk, metadata=source_metadata) for chunk in chunks]

# 🧠 Создаём эмбеддинги
embedding_model = OpenAIEmbeddings()


# Создаём ChromaDB с persistence
vectorstore = Chroma(persist_directory="chroma_store", embedding_function=embedding_model)

# ➕ Добавляем документы
vectorstore.add_documents(documents)

# 💾 Сохраняем индекс на диск
vectorstore.persist()

print(f"✅ Загружено и проиндексировано {len(documents)} чанков.")
