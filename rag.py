import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from config import INDEX_NAME, EMBEDDING_MODEL, LLM_MODEL

load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embedding_model)
llm = ChatGroq(model=LLM_MODEL, api_key=os.getenv("GROQ_API_KEY"))

prompt_template = ChatPromptTemplate.from_template("""
You are a helpful AI tutor. Answer the question based on the context below.
If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
""")

def get_chain(subject: str):
    """Returns a chain filtered to the selected subject."""
    if subject == "All Subjects":
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4, "filter": {"subject": subject}}
        )

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )