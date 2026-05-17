import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
def ingest_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    loader = PyPDFLoader(tmp_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    # Show progress per batch
    progress = st.progress(0, text="Embedding chunks...")
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        PineconeVectorStore.from_documents(
            batch,
            embedding=embedding_model,
            index_name="ai-tutor-index"
        )
        progress.progress(
            min((i + batch_size) / len(chunks), 1.0),
            text=f"Uploading chunk {min(i+batch_size, len(chunks))} of {len(chunks)}..."
        )

    progress.empty()
    os.remove(tmp_path)
    return len(chunks)

# Build retriever from existing Pinecone index
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = PineconeVectorStore(index_name="ai-tutor-index", embedding=embedding_model)
retriever = vectorstore.as_retriever()

# LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI tutor. Answer the question based on the context below.
If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
""")

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Streamlit UI
st.title("Your Personal AI Tutor")

with st.sidebar:
    st.markdown("### 📂 Upload study material")
    uploaded = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        if st.button("Add to knowledge base"):
            for file in uploaded:
                with st.spinner(f"Indexing {file.name}..."):
                    count = ingest_uploaded_file(file)
                st.success(f"✓ {file.name} — {count} chunks added")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything about the subject!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chain.stream(prompt)
        full_response = st.write_stream(response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})