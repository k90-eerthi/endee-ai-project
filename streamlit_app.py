import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("AI Semantic Search")

with open("documents.txt", "r") as f:
    documents = [d.strip() for d in f.readlines()]

query = st.text_input("Enter your search query:")

if query:
    with st.spinner("Loading..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        doc_embeddings = model.encode(documents)
        query_embedding = model.encode([query])

        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        best_index = np.argmax(similarities)

    st.write("Best Result:")
    st.write(documents[best_index])
