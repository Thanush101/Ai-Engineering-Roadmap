from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
# documents = [
#     "Climate change is a major global challenge.",
#     "Artificial intelligence is transforming industries.",
#     "Electric vehicles are the future of transportation.",
#     "Quantum computing is the next frontier in technology.",
#     "Healthcare innovation is improving patient outcomes."
# ]

llm = init_chat_model("groq:llama-3.3-70b-versatile",)

# Loading the PDF using PyPDFLoader
loader = PyPDFLoader(r"E:\Self Made Projects\Ai-Engineering-Roadmap\lessons\Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf")
documents = loader.load()

# splitting documents into chunks 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
chunks = text_splitter.split_documents(documents)

# embedding the model 
embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
# embeddings = embedding_model.encode(split_documents)
# embeddings_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# loading the embeddings into vector database
db = Chroma.from_documents(chunks, embedding=embedding_model)

query = "What is the Bishop Pattern Recognition?"
results = db.similarity_search(query, k=2)
# for res in results:
#     print(res.page_content)

# print("Results:", results)
result = llm.invoke(f"This is the user's query:{query} and {results} is the result generated. Give a proper answer to the query.")
print(result.content)