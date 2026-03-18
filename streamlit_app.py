import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Page configuration
st.set_page_config(page_title="AI Semantic Search", layout="centered")

st.title("🤖 AI Semantic Search")
st.write("Find the most relevant documents using AI")

# Load documents
try:
    with open("documents.txt", "r") as f:
        documents = [d.strip() for d in f.readlines()]
except FileNotFoundError:
    st.error("❌ documents.txt not found! Please upload it to your repo.")
    documents = []

# Input query
query = st.text_input("🔍 Enter your search query:")

if query and documents:
    with st.spinner("🔄 Processing with AI model..."):
        # Load the AI model
        model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Compute embeddings
        doc_embeddings = model.encode(documents)
        query_embedding = model.encode([query])
        
        # Compute similarity
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        top_indices = similarities.argsort()[-3:][::-1]

    st.subheader("📌 Top Results:")
    for i in top_indices:
        st.write(f"👉 {documents[i]}")
        st.caption(f"Similarity Score: {similarities[i]:.2f}")
