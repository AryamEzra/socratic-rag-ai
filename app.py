import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

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