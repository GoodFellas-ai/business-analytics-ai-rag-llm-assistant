import streamlit as st
import os
import time

from dotenv import load_dotenv

import pandas as pd

from utils import load_csv
from charts import revenue_chart, country_chart

# LangChain
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# CONFIG
st.set_page_config(
    page_title="Business Analytics AI",
    layout="wide"
)

# HERO UI
st.markdown("""
<div style="
    text-align:center;
    padding: 2rem 1rem;
    color:white;
">

  st.title("📊 Business Analytics AI Assistant")
st.subheader("Please ask some kind of questions. Then get well, trivial and professional insights and make decisions..")
st.caption("RAG + LLM + Streamlit + ChromaDB + GROQ LLM")

# ENV
app_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(app_dir, "../.env"))

# SIDEBAR
st.sidebar.title("⚙️ Controls")

uploaded_csv = st.sidebar.file_uploader("Upload CSV", type=["csv"])

df = None

if uploaded_csv:
    df = load_csv(uploaded_csv)
    st.sidebar.success("CSV Loaded")

# DASHBOARD
if df is not None:

    total_revenue = round(df["Revenue"].sum(), 2)

    top_country = df.groupby("Country")["Revenue"].sum().idxmax()

    total_customers = df["CustomerID"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric("Revenue", f"${total_revenue:,.0f}")
    col2.metric("Top Country", top_country)
    col3.metric("Customers", total_customers)

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(revenue_chart(df), use_container_width=True)

    with c2:
        st.plotly_chart(country_chart(df), use_container_width=True)

st.markdown("---")

# API CHECK
if "GROQ_API_KEY" not in os.environ:
    st.error("Missing GROQ_API_KEY")
    st.stop()

# TYPEWRITER (LEVEL 3 FEATURE)
def type_writer(text, speed=0.003):
    placeholder = st.empty()
    typed = ""

    for char in text:
        typed += char
        placeholder.markdown(f"""
        <div style="
            background:#0f172a;
            color:#00f2fe;
            padding:12px;
            border-radius:12px;
            margin-top:10px;
        ">
        🤖 {typed}
        </div>
        """, unsafe_allow_html=True)
        time.sleep(speed)

# CACHE RAG
@st.cache_resource
def load_rag():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir = os.path.join(app_dir, "../data/processed/chroma_db")

    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a business analytics assistant.\nContext:\n{context}"),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    return create_retrieval_chain(retriever, document_chain)

# INIT
if "rag_chain" not in st.session_state:
    with st.spinner("Loading AI System..."):
        st.session_state.rag_chain = load_rag()

# MEMORY
if "messages" not in st.session_state:
    st.session_state.messages = []

# CHAT DISPLAY (LEVEL 3 UI)
for msg in st.session_state.messages:

    if msg["role"] == "user":
        st.markdown(f"""
        <div style="
            background:#1f2937;
            color:white;
            padding:12px;
            border-radius:12px;
            margin:8px 0;
        ">
        🧑‍💻 {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="
            background:#0f172a;
            color:#00f2fe;
            padding:12px;
            border-radius:12px;
            margin:8px 0;
        ">
        🤖 {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

# INPUT
user_input = st.chat_input("Ask analytics questions...")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        try:
            response = st.session_state.rag_chain.invoke({
                "input": user_input
            })

            answer = response["answer"]

            # LEVEL 3 MAGIC
            type_writer(answer, speed=0.002)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(str(e))
