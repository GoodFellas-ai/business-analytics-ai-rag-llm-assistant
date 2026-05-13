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

# CONFIG
st.set_page_config(
    page_title="Business Analytics AI",
    layout="wide"
)

# 👇 
st.markdown("""
<div style="
    padding: 2rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #1f1f2e, #2b2b40);
    color: white;
    text-align: center;
    box-shadow: 0px 8px 24px rgba(0,0,0,0.25);
    margin-bottom: 1.5rem;
">

    <h1 style="margin-bottom: 0.5rem; font-size: 2.2rem;">
        📊 Business Analytics AI Assistant
    </h1>

    <p style="font-size: 1.1rem; opacity: 0.9;">
        Upload your sales data and ask questions about your business
    </p>

    <p style="font-size: 0.9rem; opacity: 0.7;">
        AI-powered RAG system using Streamlit • ChromaDB • GROQ LLM
    </p>

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
