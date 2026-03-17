import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("AI Semantic Search")

# Load documents
with open("documents.txt", "r") as f:
    documents = f.readlines()

documents = [d.strip() for d in documents]

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

doc_embeddings = model.encode(documents)

query = st.text_input("Enter your search query")

if query:
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    best_index = np.argmax(similarities)

    st.subheader("Best Result:")
    st.write(documents[best_index])
