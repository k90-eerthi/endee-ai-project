from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load documents
with open("documents.txt", "r") as file:
    documents = file.readlines()

documents = [doc.strip() for doc in documents]

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert documents to embeddings
doc_embeddings = model.encode(documents)

# Ask user query
query = input("Enter your search query: ")

# Convert query to embedding
query_embedding = model.encode([query])

# Calculate similarity
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

# Get best result
best_index = np.argmax(similarities)

print("\nMost Relevant Document:")
print(documents[best_index])
