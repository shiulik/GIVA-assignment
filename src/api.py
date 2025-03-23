from fastapi import FastAPI, UploadFile, File
import os
from src.faiss_index import FaissIndex
from src.embeddings import generate_embedding, load_documents

DATA_FOLDER = "data/"
INDEX_PATH = "index/vector_index.faiss"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs("index/", exist_ok=True)

app = FastAPI()

documents = load_documents(DATA_FOLDER)
document_embeddings = [generate_embedding(doc) for doc in documents] if documents else []

embedding_dim = len(document_embeddings[0]) if document_embeddings else 384
faiss_index = FaissIndex(dimension=embedding_dim, index_path=INDEX_PATH)

if document_embeddings:
    faiss_index.add_embeddings(document_embeddings)
    faiss_index.save_index(INDEX_PATH)

@app.get("/")
def home():
    return {"message": "Welcome to the Document Similarity Search API"}

@app.post("/api/add_document/")
async def add_document(file: UploadFile = File(...)):
    """
    API to upload a document and update the FAISS index.
    """
    file_path = os.path.join(DATA_FOLDER, file.filename)

    content = await file.read()  
    text = content.decode("utf-8").strip()

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    new_embedding = generate_embedding(text)

    documents.append(text)
    faiss_index.add_embeddings([new_embedding])
    faiss_index.save_index(INDEX_PATH)

    return {"message": "Document added successfully!", "filename": file.filename}

@app.get("/api/search/")
def search(query: str):
    """
    API to search for similar documents and return top 5 matches.
    """
    if not documents:
        return {"error": "No documents found! Please upload documents first."}

    query_embedding = generate_embedding(query)

    top_k = 5
    indices = faiss_index.search(query_embedding, top_k)

    print(f"Documents Count: {len(documents)}")
    print(f"Returned Indices: {indices}")

    valid_results = [
        {"document": documents[i], "index": i}
        for i in indices if i < len(documents)
    ]

    if not valid_results:
        return {"error": "No valid results found!"}

 
    return {
        "query": query,
        "results": [result["document"] for result in valid_results]
    }
