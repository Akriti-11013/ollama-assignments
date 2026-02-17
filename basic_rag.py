from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# LLM
llm = OllamaLLM(model="phi3")

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Data
docs = [
    "RAG stands for Retrieval Augmented Generation",
    "LangChain is a framework for LLM-based apps",
    "Ollama runs LLMs locally"
]

# Split
splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=10)
texts = splitter.split_text(" ".join(docs))

# Vector store
db = FAISS.from_texts(texts, embeddings)

# Query
docs = db.similarity_search("What is RAG?")
for d in docs:
    print(d.page_content)

# Simple RAG answer
context = " ".join([d.page_content for d in docs])
response = llm.invoke(f"Answer using context:\n{context}\n\nQuestion: What is RAG?")
print("\nAnswer:", response)
