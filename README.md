💠 Enterprise Operational Intelligence Assistant (RAG System)

📌 Overview

In today's data-driven world, enterprises generate massive amounts of operational data (e.g., consumer complaints, incident reports, internal logs). However, querying this data efficiently while maintaining strict data privacy is a major challenge.

This project is an End-to-End Enterprise RAG (Retrieval-Augmented Generation) System. It allows users to ask natural language questions about internal operational data and receive highly accurate answers backed by verified sources.

📊 Tableau Business Intelligence Dashboard

To provide executive-level insights, the cleaned operational data is visualized using Tableau. The dashboard tracks the geographical distribution of complaints, temporal trends, and highlights the companies/products with the highest volume of issues.

🌟 Key Features

🔒 100% Offline & Private: The entire system runs locally. No sensitive enterprise data is sent to external APIs (like OpenAI), ensuring complete data privacy.

🛡️ Zero-Hallucination Guardrail: Built with custom pre-generation relevance checks. If the answer does not exist in the retrieved data, the AI is explicitly blocked from hallucinating and safely responds with "I cannot answer this based on the provided data."

🎯 Verifiable Sources: Every AI-generated answer is accompanied by the exact source documents (complaint narratives, product, company) used to formulate the response, building trust in the system.

💻 Premium UI: A responsive, dark-mode compatible web interface built with Streamlit for a smooth user experience.

🏗️ Architecture & Tech Stack

Data Pipeline: Pandas (Data cleaning, handling missing values, feature engineering).

Embedding Model: sentence-transformers/all-MiniLM-L6-v2 (Fast, local vectorization).

Vector Database: ChromaDB (Efficient document retrieval).

LLM Engine: google/flan-t5-base (Local, lightweight model fine-tuned for strict instruction following).

Orchestration: LangChain (Connecting the retriever with the LLM).

Frontend: Streamlit (Interactive web application).

🚀 How to Run the Project Locally

1. Clone the repository

git clone [https://github.com/Ahmed-abbas-cs/Operational-Intelligence-RAG.git](https://github.com/Ahmed-abbas-cs/Operational-Intelligence-RAG.git)
cd Operational-Intelligence-RAG


2. Create a virtual environment & install dependencies

python3 -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install pandas langchain langchain-community chromadb sentence-transformers transformers streamlit


3. Run the Data Cleaning Script
Make sure you have your raw complaints.csv in the root folder, then run:

python data_cleaning.py


(This will generate a cleaned_complaints_data.csv file optimized for the RAG engine).

4. Launch the AI Assistant

streamlit run rag_engine.py


🧪 Testing the Anti-Hallucination Guardrail

To verify the enterprise-grade safety of this system, try asking the deployed app:

"How can I fix my broken iPhone screen?"

Expected Output: The system will instantly block the request and reply: I cannot answer this based on the provided data. (Proving the guardrail works!).

👨‍💻 Author

Built by a Data Engineer passionate about turning raw data into safe, actionable AI intelligence.
