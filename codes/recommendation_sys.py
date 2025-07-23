from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Example items
items = [
    "Deep learning for image recognition.",
    "Healthy recipes for breakfast.",
    "Understanding neural networks.",
    "Yoga and mindfulness practices.",
    "Learning AI"
]

# User profile (could be a single string or aggregation of past liked items)
user_profile = "I want to learn about artificial intelligence and neural networks."

# Encode items and user profile
model = SentenceTransformer('all-MiniLM-L6-v2')
item_embeddings = model.encode(items)
user_embedding = model.encode([user_profile])

# Compute similarity
scores = cosine_similarity(user_embedding, item_embeddings)[0]
top_indices = np.argsort(scores)[::-1]  # Descending order

# Recommend top 2 items
for idx in top_indices[:2]:
    print(f"Recommended: {items[idx]} (Score: {scores[idx]:.2f})")

