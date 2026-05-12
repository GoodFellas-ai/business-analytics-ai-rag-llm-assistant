# BUSINESS ANALYTICS AI - LLM - RAG  ASSISTANT


Upload your sales data and ask questions about your business! This AI-LLM-RAG application deployed on Streamlit, using ChromaDB for vector storage and GROQ for LLM responses.
business-analytics-ai-rag-llm-assistant
AI-powered Business Analytics Assistant using RAG (Retrieval-Augmented Generation), LLMs, and Vector Databases for intelligent business insight generation from retail sales data.

----

## 🚀 Project Overview

This project is an end-to-end Business Intelligence + AI application that allows users to analyze sales data using natural language questions.
The system combines:
Business Analytics
RAG Architecture
LLM-based Question Answering
Vector Search


## Interactive Dashboarding

Users can ask questions such as:
"Which country generated the highest revenue?"
"What were the monthly sales trends?"
"Who are the top customers?"
"Which products performed best?"
The application retrieves relevant business insights from a vector database and generates contextual responses using an LLM.

## 🧠 Technologies Used

AI / LLM / RAG
Llama 3.1 (via GROQ API)
LangChain
ChromaDB
Sentence Transformers
RAG Pipeline Architecture
Data Science & Analytics
Python
Pandas
NumPy
Matplotlib
Plotly
Web Application
Streamlit
Vector Embeddings
all-MiniLM-L6-v2


## 📊 Features

Interactive Business Analytics Dashboard
Natural Language Question Answering
KPI Monitoring
Revenue Trend Analysis
Customer & Product Insights
Vector Similarity Search
Bilingual Response Support (English & Turkish)
Retrieval-Augmented Generation (RAG)


## 🗂️ Project Structure

 Bash

business-analytics-ai-rag-llm-assistant/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_cleaning_eda.ipynb
│   ├── 02_business_metrics.ipynb
│   └── 03_rag_preparation.ipynb
│
├── requirements.txt
├── README.md
└── .env

----

## ⚙️ Installation

Clone the repository:
Bash
git clone <your-repo-link>
cd business-analytics-ai-rag-llm-assistant
Create a virtual environment:
Bash
python -m venv .venv
Activate the environment:
Windows
Bash
.venv\Scripts\activate


Install dependencies:
Bash
pip install -r requirements.txt


## 🔑 Environment Variables

Create a .env file:
Plain text
GROQ_API_KEY=your_api_key_here


## ▶️ Run the Application
Bash
streamlit run app/app.py


## 📈 Dataset
# Dataset used:

Online Retail Dataset
The dataset contains transactional retail sales records including:
Invoice information
Products
Quantities
Customer IDs
Countries
Revenue data


## 🧩 RAG Workflow

Plain text
Business Data
→ Data Cleaning & Analytics
→ Insight Summaries
→ Text Embeddings
→ ChromaDB Vector Storage
→ Similarity Retrieval
→ LLM Response Generation

## 🌍 Deployment

The application is designed for deployment on:
Hugging Face Spaces
Streamlit Cloud

## 🎯 Future Improvements

Advanced KPI dashboards
RAG evaluation with RAGAS
FAISS support
CI/CD pipelines
Multi-dataset support
Advanced business forecasting
SQL agent integration

## 👨‍💻 Author

Developed by Erdal Goodman as part of an AI, Data Science and Business Intelligence portfolio project.