import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Load book data from a CSV file (like a spreadsheet)
# We only want columns 1, 2, and 4 (Book Title, Author, Publisher)
df = pd.read_csv('Books.csv',usecols=[ 1,2,4])

# print(df.head(100))

# Take only the first 100 books to make processing faster
items = df.head(100)


# What kind of books does the user like?
# This is like telling a librarian "I like these types of books"
user_profile = "science fiction"

# user_profile = """
# I primarily enjoy science fiction (especially space opera), 
# but I also like some fantasy (urban fantasy preferred), 
# and I occasionally read mystery novels. 
# I avoid romance and horror genres.
# """




# Load the AI model that understands text meaning
# Think of this as a smart translator that converts words into numbers
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#all-MiniLM-L6-v2
model = SentenceTransformer('all-mpnet-base-v2')



# Prepare book data for analysis

# Combine all text features into a single string
items_text = items.apply(lambda x: f"{x.iloc[0]} by {x.iloc[1]} published by {x.iloc[2]}", axis=1).tolist()

# Use only book titles for similarity
# items_text = items.iloc[:, 0].tolist()  # First column (Book-Title)


# # Combine title and author
# items_text = items.apply(lambda x: f"{x.iloc[0]} by {x.iloc[1]}", axis=1).tolist()

# Convert book titles into numbers the computer can understand
# The AI reads each book title and creates a "fingerprint" of numbers
item_embeddings = model.encode(items_text)



# Convert the user's preferences into the same type of numbers
# Now we can compare what the user likes with the books
user_embedding = model.encode([user_profile])



# Calculate how similar each book is to what the user likes
# This is like asking "how close is each book to what this person enjoys?"
scores = cosine_similarity(user_embedding, item_embeddings)[0]



# Sort books from most similar to least similar
# Find the books that match best with user's taste
top_indices = np.argsort(scores)[::-1]  # Arrange from highest to lowest score

# Show the top 2 book recommendations
# These are the books most likely to match what the user enjoys
for idx in top_indices[:2]:
    print(f"Recommended: {items_text[idx]} (Score: {scores[idx]:.2f})")

