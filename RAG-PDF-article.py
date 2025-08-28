"""
File: RAG-PDF-article.py
Author: Andrew Stollak
Date: 2025-08-26
Description: This script demonstrates RAG on local LM Studio server running mistral.
    Specifically, mistralai/mistral-7b-instruct-v0.3 on LM Studio LM Studio 0.3.23 (Build 3)

License: MIT
Version: 0.1.0
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings.huggingface import HuggingFaceEmbeddings # Or a compatible LM Studio embedding model
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr

pdf_file = "2504.15585v4.pdf"

# Load & chunk PDF article
article = PyPDFLoader(pdf_file).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(article)

# Map chunks to embeddings and store both in ChromaDB
embedding_vectors = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # Example
embedding_db = Chroma.from_documents(chunks, embedding_vectors, persist_directory="./chroma_db")
embedding_retriever = embedding_db.as_retriever()

# Initialize LM Studio LLM
# api_key wants type None or SecretStr,
#   but None default to the environmental variable
#   which is a problem, so give it nonsense
llm = ChatOpenAI(base_url="http://localhost:1234/v1",
                 api_key=SecretStr("none-needed"),
                 model="mistral") # Adjust URL and model name

# RAG chain is made of the RAG DB plus the document chain,
# which is made of the LM (LLM, SLM) and the prompt.
# This version of Mistral, mistralai/mistral-7b-instruct-v0.3b,
# takes an "assistant" and a "user" prompt
# Other models need other types
prompt = ChatPromptTemplate.from_messages([
    ("assistant", "You are a useful AI research assistant. Answer questions based on provided context ."),
    ("user", "Context: {context}\nQuestion: {input}")
])
doc_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(embedding_retriever, doc_chain)

# Console chat
print(f"Loaded {pdf_file}")
print("Good starting question: What are the some security considerations for LLMs?")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit", "bye", "adios"]:
        break
    response = rag_chain.invoke({"input": query})
    print(f"AI: {response['answer']}")
