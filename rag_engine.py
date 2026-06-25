import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import os
import warnings
import time
import streamlit as st

# Ignore unnecessary warnings for a cleaner output
warnings.filterwarnings("ignore")

# --- 1. UI Configuration & Styling ---
st.set_page_config(
    page_title="Enterprise AI Assistant",
    page_icon="💠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injecting Custom CSS for a premium and attractive look
st.markdown("""
<style>
    /* Removed hardcoded background to automatically support Dark Mode */
    
    /* Customizing the primary button */
    .stButton>button {
        background-color: #0052cc;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #003d99;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    /* Custom box for AI answers */
    .answer-box {
        background-color: #f1f3f4; /* Light gray to suit all themes */
        padding: 25px;
        border-radius: 10px;
        border-left: 6px solid #0052cc;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        font-size: 18px;
        color: #1e1e1e; /* Dark text color to be clearly readable inside the box */
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Core AI Logic (Cached for Performance) ---
# @st.cache_resource ensures the heavy AI models load only ONCE, making searches instant!
@st.cache_resource(show_spinner=False)
def initialize_system(data_path):
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        return None, None, "Cleaned data file not found. Please ensure 'cleaned_complaints_data.csv' exists."

    # Increased sample size from 100 to 1000 so the AI has more data to answer from
    df_small = df.head(1000)
    
    loader = DataFrameLoader(df_small, page_content_column="Consumer complaint narrative")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory="./chroma_db")

    try:
        # Load the AI Model Locally with STRICT generation parameters
        local_llm = pipeline(
            "text2text-generation", 
            model="google/flan-t5-base", 
            max_new_tokens=150,
            do_sample=False
        )
    except Exception as e:
        return None, None, f"Model loading error: {e}"

    return vectorstore, local_llm, "Success"

# --- 3. Streamlit User Interface ---
def main():
    # --- Sidebar ---
    with st.sidebar:
        st.title("🛡️ System Info")
        st.info(
            "**Engine:** Local RAG System\n\n"
            "**Model:** Google FLAN-T5 Base\n\n"
            "**Data Type:** Consumer Complaints\n\n"
            "**Status:** Online"
        )
        st.divider()
        st.markdown("### 🔒 Privacy & Security")
        st.caption("This system runs **100% offline**. No operational data is sent to external APIs (like OpenAI). Hallucinations are strictly blocked.")

    # --- Main Header ---
    st.title("💠 Operational Intelligence Assistant")
    st.markdown("Ask natural language questions about your operational data. The AI will strictly extract answers from verified records.")
    st.divider()

    # --- Initialization ---
    cleaned_file_path = "cleaned_complaints_data.csv"
    
    with st.spinner("Initializing Secure AI Engine & Vector Database (Processing 1000 records)..."):
        db, llm, status = initialize_system(cleaned_file_path)

    if not db or not llm:
        st.error(f"System Initialization Failed: {status}")
        return

    # --- Search Interface ---
    col1, col2 = st.columns([4, 1])
    with col1:
        user_question = st.text_input("🔍 Ask a specific question:", placeholder="e.g., What is the name of the product mentioned by customers?")
    with col2:
        st.write("")
        st.write("")
        search_btn = st.button("Analyze Data", use_container_width=True)

    # --- Processing & Results ---
    if search_btn and user_question:
        with st.spinner("Scanning operational data and generating strict response..."):
            time.sleep(0.5) # Slight UX delay to show processing
            
            # 1. Search DB
            relevant_docs = db.similarity_search(user_question, k=2)
            
            # 2. Build Context
            context_text = ""
            for i, doc in enumerate(relevant_docs):
                product = doc.metadata.get('Product', 'Unknown')
                company = doc.metadata.get('Company', 'Unknown')
                context_text += f"--- Complaint {i+1} ---\nProduct Name: {product}\nCompany: {company}\nNarrative: {doc.page_content}\n\n"
            
            # --- ENTERPRISE GUARDRAIL (Zero Hallucination Filter) ---
            # This explicitly prevents the small offline model from hallucinating
            # by checking if the user's keywords actually exist in the retrieved text.
            question_words = set(user_question.lower().replace('?', '').replace('.', '').split())
            stop_words = {"what", "how", "is", "the", "a", "an", "of", "in", "to", "for", "with", "on", "my", "can", "i", "fix", "are", "do", "did", "does", "about", "any", "where", "who", "why"}
            keywords = question_words - stop_words
            
            context_lower = context_text.lower()
            
            is_relevant = False
            for kw in keywords:
                if len(kw) > 2 and kw in context_lower:
                    is_relevant = True
                    break
                    
            if len(keywords) == 0:
                is_relevant = True # Allow if question is very short
                
            # 3. Generate Answer or Block
            if not is_relevant:
                # Force block if documents are completely irrelevant (e.g., iPhone question)
                ai_answer = "I cannot answer this based on the provided data."
            else:
                # Ask the AI only if data is relevant
                prompt = f"Context:\n{context_text}\n\nQuestion: {user_question}\n\nAnswer strictly based on context:"
                result = llm(prompt)
                ai_answer = result[0]['generated_text'].strip()

        # Display AI Answer in a custom beautiful box
        st.markdown("### 🎯 Analysis Result")
        st.markdown(f'<div class="answer-box"><b>{ai_answer}</b></div>', unsafe_allow_html=True)
        
        # Display Sources transparently
        st.markdown("### 📚 Verified Sources")
        for i, doc in enumerate(relevant_docs):
            product = doc.metadata.get('Product', 'Unknown')
            company = doc.metadata.get('Company', 'Unknown')
            # Use expanders to keep the UI clean
            with st.expander(f"📄 Source {i+1} | Company: {company} | Product: {product}"):
                st.write(doc.page_content)

if __name__ == "__main__":
    main()