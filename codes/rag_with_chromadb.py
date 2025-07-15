from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# documents = [
#     "Climate change is a major global challenge.",
#     "Artificial intelligence is transforming industries.",
#     "Electric vehicles are the future of transportation.",
#     "Quantum computing is the next frontier in technology.",
#     "Healthcare innovation is improving patient outcomes."
# ]

loader = PyPDFLoader(r"E:/Self Made Projects/Ai-Engineering-Roadmap/lessons/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
split_documents = text_splitter.split_documents(documents)

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(split_documents)
# embeddings_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = Chroma.from_texts(split_documents, embedding_model)

query = "What is the future of technology?"
results = db.similarity_search(query, k=2)
for res in results:
    print(res.page_content)
