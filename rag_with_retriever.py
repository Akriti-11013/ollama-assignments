from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# 1️⃣ LLM
llm = OllamaLLM(model="phi3")

# 2️⃣ Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3️⃣ Data
documents = [
    "RAG stands for Retrieval Augmented Generation knowledges are retrieved before answering",
    "LangChain is a framework to build LLM applications",
    "FAISS is used for vector similarity search",
    "Ollama allows running LLMs locally"
]

# 4️⃣ Split
splitter = CharacterTextSplitter(chunk_size=60, chunk_overlap=10)
texts = splitter.split_text(" ".join(documents))

# 5️⃣ Vector DB
db = FAISS.from_texts(texts, embeddings)

# 6️⃣ Retriever (THIS IS IMPORTANT)
retriever = db.as_retriever(search_kwargs={"k": 2})

# 7️⃣ Prompt
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Use the context below to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
)

# 8️⃣ RAG function (manual chain)
def rag_chain(question: str):
    docs = retriever.invoke(question)   # ✅ NEW WAY
    context = " ".join(doc.page_content for doc in docs)

    final_prompt = prompt.format(
        context=context,
        question=question
    )

    return llm.invoke(final_prompt)


# 9️⃣ Ask question
response = rag_chain("What is RAG?")
print("\nRAG Answer:\n", response)
