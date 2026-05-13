import streamlit as st
import os

from dotenv import load_dotenv

import pandas as pd

from utils import load_csv
from charts import revenue_chart, country_chart

# LangChain
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

st.markdown("""
<style>
.hero {
    text-align: center;
    padding: 3.2rem 2rem;
    border-radius: 24px;
    background: radial-gradient(circle at top, #0f172a, #020617);
    color: white;
    box-shadow: 0 20px 60px rgba(0,0,0,0.55);
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
}

/* BIG TITLE (LEVEL 2 UPGRADE) */
.hero h1 {
    font-size: 4rem;
    font-weight: 800;
    margin-bottom: 0.6rem;
    background: linear-gradient(90deg, #4facfe, #00f2fe, #38bdf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: 1px;
}

/* subtitle */
.hero p {
    font-size: 1.25rem;
    margin: 0.4rem 0;
    opacity: 0.85;
}

/* tech line */
.hero .sub {
    font-size: 0.95rem;
    opacity: 0.65;
    margin-bottom: 1.5rem;
}

/* badges row (LEVEL 2 glass style) */
.badges {
    margin-top: 1.2rem;
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
}

.badge {
    padding: 7px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(10px);
    transition: 0.3s;
}

.badge:hover {
    transform: scale(1.05);
    background: rgba(255,255,255,0.12);
}
</style>

<div class="hero">

    <h1>📊 Business Analytics AI</h1>

    <p>Ask questions. Get insights. Make decisions.</p>

    <p class="sub">RAG-powered analytics assistant built with Streamlit + ChromaDB + GROQ LLM</p>

    <div class="badges">
        <span class="badge">🤖 AI Assistant</span>
        <span class="badge">🧠 RAG System</span>
        <span class="badge">📦 Vector DB</span>
        <span class="badge">⚡ GROQ LLM</span>
    </div>

</div>
""", unsafe_allow_html=True)

# ENV
app_dir = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(app_dir, "../.env"))

# SIDEBAR
st.sidebar.title("⚙️ Controls")

uploaded_csv = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# LOAD DATA
df = None

if uploaded_csv:

    df = load_csv(uploaded_csv)

    st.sidebar.success("CSV Loaded")

# DASHBOARD
if df is not None:

    total_revenue = round(df["Revenue"].sum(), 2)

    top_country = (
        df.groupby("Country")["Revenue"]
        .sum()
        .idxmax()
    )

    total_customers = (
        df["CustomerID"].nunique()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Revenue",
            f"${total_revenue:,.0f}"
        )

    with col2:
        st.metric(
            "Top Country",
            top_country
        )

    with col3:
        st.metric(
            "Customers",
            total_customers
        )

    # CHARTS
    chart1, chart2 = st.columns(2)

    with chart1:
        st.plotly_chart(
            revenue_chart(df),
            use_container_width=True
        )

    with chart2:
        st.plotly_chart(
            country_chart(df),
            use_container_width=True
        )

st.markdown("---")

# API CHECK
if "GROQ_API_KEY" not in os.environ:

    st.error("Missing GROQ_API_KEY")
    st.stop()

# CACHE
@st.cache_resource
def load_rag():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_dir = os.path.normpath(
        os.path.join(
            app_dir,
            "../data/processed/chroma_db"
        )
    )

    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a business analytics assistant.

Use context carefully.

Context:
{context}
"""
            ),
            ("human", "{input}")
        ]
    )

    document_chain = create_stuff_documents_chain(
        llm,
        prompt
    )

    retriever = vector_db.as_retriever(
        search_kwargs={"k": 5}
    )

    return create_retrieval_chain(
        retriever,
        document_chain
    )

# INIT
if "rag_chain" not in st.session_state:

    with st.spinner("Loading AI System..."):

        st.session_state.rag_chain = load_rag()

# MEMORY
if "messages" not in st.session_state:

    st.session_state.messages = []

# SHOW CHAT
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

# INPUT
user_input = st.chat_input(
    "Ask analytics questions..."
)

if user_input:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):

        st.markdown(user_input)

    with st.chat_message("assistant"):

        try:

            response = (
                st.session_state.rag_chain.invoke(
                    {
                        "input": user_input
                    }
                )
            )

            answer = response["answer"]

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            st.error(str(e))
