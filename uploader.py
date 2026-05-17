import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import INDEX_NAME
from rag import embedding_model


def ingest_uploaded_file(uploaded_file, subject: str) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    for doc in documents:
        doc.metadata["subject"] = subject

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    progress = st.progress(0, text="Embedding chunks...")
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        PineconeVectorStore.from_documents(
            batch,
            embedding=embedding_model,
            index_name=INDEX_NAME,
        )
        progress.progress(
            min((i + batch_size) / len(chunks), 1.0),
            text=f"Uploading chunk {min(i + batch_size, len(chunks))} of {len(chunks)}...",
        )

    progress.empty()
    os.remove(tmp_path)
    return len(chunks)
import os
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag import embedding_model
from config import INDEX_NAME

def ingest_uploaded_file(uploaded_file, subject: str) -> int:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Tag every chunk with subject so we can filter later
    for doc in documents:
        doc.metadata["subject"] = subject

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    progress = st.progress(0, text="Embedding chunks...")
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        PineconeVectorStore.from_documents(
            batch,
            embedding=embedding_model,
            index_name=INDEX_NAME
        )
        progress.progress(
            min((i + batch_size) / len(chunks), 1.0),
            text=f"Uploading chunk {min(i + batch_size, len(chunks))} of {len(chunks)}..."
        )

    progress.empty()
    os.remove(tmp_path)
    return len(chunks)