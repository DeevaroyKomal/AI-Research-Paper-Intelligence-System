import streamlit as st
import pandas as pd
import numpy as np
import os
import faiss
import time
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from keybert import KeyBERT

# Set Page Config
st.set_page_config(
    page_title="AI Research Paper Inteligence System",
    page_icon="🔍",
    layout="wide"
)

# Streamlit Resource Caching
@st.cache_resource(show_spinner=False)
def load_resources():
    # 1. Load dataset metadata
    parquet_path = "papers_metadata.parquet"
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    else:
        dataset = load_dataset("CShorten/ML-ArXiv-Papers", split='train')
        df = pd.DataFrame(dataset)
        df = df[['title', 'abstract']].head(15000)
        df.to_parquet(parquet_path)
        
    # 2. Initialize SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 3. Load/Generate Embeddings
    embedding_path = "embedding.npy"
    if os.path.exists(embedding_path):
        embedding = np.load(embedding_path)
    else:
        # Pre-process text field
        df["paper_text"] = df["title"] + " " + df["abstract"]
        df["paper_text"] = df["paper_text"].str.replace("\n", " ", regex=False).str.strip()
        embedding = model.encode(df["paper_text"].tolist(), batch_size=32, show_progress_bar=False)
        np.save(embedding_path, embedding)
        
    # 4. Build FAISS index for L2 inner product (cosine similarity once normalized)
    faiss.normalize_L2(embedding)
    index = faiss.IndexFlatIP(384)
    index.add(embedding)
    
    # 5. Load BART Summarizer
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    # 6. Load KeyBERT
    kw_model = KeyBERT(model=model)
    
    return df, model, index, summarizer, kw_model

# Load resources with spinner
try:
    with st.spinner("Loading models, embeddings, and dataset... This may take a minute on the first run."):
        df, model, index, summarizer, kw_model = load_resources()
    st.success("All AI models and index loaded successfully!")
except Exception as e:
    st.error(f"Error loading system resources: {e}")
    st.stop()

# Title
st.title("AI Research Paper Inteligence System")
st.write(
    "Enter a machine learning research topic below to find the most relevant papers, "
    "generate summaries, and extract keywords."
)

# Search form
with st.container():
    query = st.text_input("Enter Research Topic/Query:", placeholder="e.g. deep learning for medical image analysis")
    search_clicked = st.button("Search", type="primary")

if search_clicked or query:
    if not query.strip():
        st.warning("Please enter a research topic to search.")
    else:
        st.markdown("---")
        st.markdown(f"### Research Topic: *\"{query}\"*")
        
        with st.spinner("Searching and processing..."):
            # Compute query embedding
            query_embedding = model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Semantic search
            D, I = index.search(query_embedding, 5)
            
            # --- BEST MATCHING PAPER (Result #1) ---
            best_idx = int(I[0][0])
            best_score = float(D[0][0])
            best_title = df.iloc[best_idx]["title"]
            best_abstract = df.iloc[best_idx]["abstract"]
            
            # Summarization
            try:
                summary_output = summarizer(best_abstract, max_length=120, min_length=40)
                summary_text = summary_output[0]["summary_text"]
            except Exception as e:
                summary_text = f"Could not generate summary: {e}"
                
            # Keyphrase extraction
            try:
                # Fix: Passing the best matching paper's abstract instead of a global variable
                keywords = kw_model.extract_keywords(
                    best_abstract, 
                    keyphrase_ngram_range=(1, 3), 
                    stop_words="english"
                )
                keyword_tags = [kw[0] for kw in keywords]
            except Exception as e:
                keyword_tags = []
                
            # Render best matching paper
            st.markdown("#### 🏆 Best Matching Paper")
            best_card = st.container(border=True)
            with best_card:
                st.subheader(best_title)
                st.markdown(f"**Similarity Score:** `{best_score * 100:.2f}% Match`")
                st.write("**Abstract:**")
                st.write(best_abstract)
                
                st.info(f"**🤖 AI-Generated Summary (BART):**\n\n{summary_text}")
                
                if keyword_tags:
                    st.write("**Extracted Keywords (KeyBERT):**")
                    tags_md = " ".join([f"`{kw}`" for kw in keyword_tags])
                    st.markdown(tags_md)
                    
            # --- REMAINING TOP-4 SIMILAR PAPERS ---
            st.markdown("#### 📂 Other Similar Papers")
            for j in range(1, 5):
                idx = int(I[0][j])
                score = float(D[0][j])
                title = df.iloc[idx]["title"]
                abstract = df.iloc[idx]["abstract"]
                
                with st.expander(f"Paper #{j+1}: {title} (Match: {score * 100:.2f}%)"):
                    st.write(f"**Abstract:** {abstract}")
