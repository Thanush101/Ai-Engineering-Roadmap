from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Example data
texts = [
    "Football match tonight.",
    "Stock market crashes.",
    "New AI research published.",
    "The team won the championship.",
    "Basketball league starts next week.",
    "Investors are worried about inflation.",
    "Tech conference to showcase new gadgets.",
    "Tennis player wins grand slam title.",
    "Cryptocurrency prices surge overnight.",
    "New smartphone model breaks sales record.",
    "Hockey finals draw huge crowds.",
    "Central bank announces interest rate hike.",
    "Scientists develop faster computer chips.",
    "Local football club signs new striker.",
    "Bank profits rise despite economic slowdown.",
    "Breakthrough in quantum computing.",
    "Baseball season opens with thrilling games.",
    "Real estate stocks gain momentum.",
    "Startup launches innovative AI product.",
    "Star athlete announces retirement from sports."
]

labels = [
    "sports",
    "finance",
    "tech",
    "sports",
    "sports",
    "finance",
    "tech",
    "sports",
    "finance",
    "tech",
    "sports",
    "finance",
    "tech",
    "sports",
    "finance",
    "tech",
    "sports",
    "finance",
    "tech",
    "sports"
]


# Encode texts
model = SentenceTransformer('all-MiniLM-L6-v2')
X = model.encode(texts)

# Train-test split
#When you split your data into train and test sets using train_test_split and shuffle=True (which is the default), the data is randomly shuffled before splitting.
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)

# Random Forest classifier
#n_estimators=100 means "let’s create 100 such trees and combine their votes."
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
