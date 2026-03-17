from sentence_transformers import SentenceTransformer
import numpy as np

# Load documents
with open("documents.txt","r") as f:
    documents = f.readlines()

documents = [d.strip() for d in documents]

# Load AI model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Convert documents to vectors
doc_vectors = model.encode(documents)

query = input("Enter your search: ")

query_vector = model.encode([query])

similarities = np.dot(doc_vectors, query_vector.T)

best_match = np.argmax(similarities)

print("Best Result:")
print(documents[best_match])