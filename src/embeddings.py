from sentence_transformers import SentenceTransformer
import os

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embedding(text):
    """
    Generates embedding for a given text.
    """
    return model.encode(text).tolist()

def load_documents(data_folder):
    """
    Loads all text files from the 'data/' directory.
    """
    documents = []
    for file_name in os.listdir(data_folder):
        file_path = os.path.join(data_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            documents.append(f.read().strip())
    return documents
