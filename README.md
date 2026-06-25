# 💠 Enterprise Operational Intelligence Assistant (RAG System)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FLAN--T5-yellow.svg)

## 📌 Overview
In today's data-driven world, enterprises generate massive amounts of operational data (e.g., consumer complaints, incident reports, internal logs). However, querying this data efficiently while maintaining strict data privacy is a major challenge. 

This project is an **End-to-End Enterprise RAG (Retrieval-Augmented Generation) System**. It allows users to ask natural language questions about internal operational data and receive highly accurate answers backed by verified sources.

## 🌟 Key Features

* **🔒 100% Offline & Private:** The entire system runs locally. No sensitive enterprise data is sent to external APIs (like OpenAI), ensuring complete data privacy.
* **🛡️ Zero-Hallucination Guardrail:** Built with custom pre-generation relevance checks. If the answer does not exist in the retrieved data, the AI is explicitly blocked from hallucinating and safely responds with *"I cannot answer this based on the provided data."*
* **🎯 Verifiable Sources:** Every AI-generated answer is accompanied by the exact source documents (complaint narratives, product, company) used to formulate the response, building trust in the system.
* **📊 Business Intelligence Integration:** Data was pre-cleaned using Pandas and can be seamlessly connected to **Tableau** for high-level visual KPI tracking.
* **💻 Premium UI:** A responsive, dark-mode compatible web interface built with Streamlit for a smooth user experience.

## 🏗️ Architecture & Tech Stack

1. **Data Pipeline:** `Pandas` (Data cleaning, handling missing values, feature engineering).
2. **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (Fast, local vectorization).
3. **Vector Database:** `ChromaDB` (Efficient document retrieval).
4. **LLM Engine:** `google/flan-t5-base` (Local, lightweight model fine-tuned for strict instruction following).
5. **Orchestration:** `LangChain` (Connecting the retriever with the LLM).
6. **Frontend:** `Streamlit` (Interactive web application).

## 🚀 How to Run the Project Locally

**1. Clone the repository**
```bash
git clone [https://github.com/Ahmed-abbas-cs/Operational-Intelligence-RAG.git](https://github.com/Ahmed-abbas-cs/Operational-Intelligence-RAG.git)
cd Operational-Intelligence-RAG
<img width="1410" height="1188" alt="Operational Intelligence Dashboard" src="https://github.com/user-attachments/assets/d811aeef-d8b6-413d-9ef3-4b8d866a9239" />
<img width="1440" height="900" alt="Screenshot 1448-01-10 at 3 40 28 AM" src="https://github.com/user-attachments/assets/ddef488e-d932-4827-8202-cce36b8d33f2" />
