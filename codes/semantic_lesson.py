from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load a pre-trained embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Example corpus and query
documents = [
    "Machine learning is fascinating.",
    "I love playing football.",
    "Artificial intelligence and machine learning are related field",
    "Cooking is an art."
]
query = "When is sunday ?"

# Encode documents and query
doc_embeddings = model.encode(documents)
query_embedding = model.encode([query])

# Compute cosine similarity
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

# Get top match
top_idx = np.argmax(similarities)
print(f"Best match: {documents[top_idx]} (Score: {similarities[top_idx]:.2f})")

