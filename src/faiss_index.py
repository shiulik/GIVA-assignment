import faiss
import numpy as np
import os

class FaissIndex:
    def __init__(self, dimension, index_path=None):
        """
        Initializes a FAISS index.
        If index file exists, it loads the index.
        """
        self.dimension = dimension
        self.index_path = index_path

        if index_path and os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"🔹 FAISS index loaded from: {index_path}")
        else:
            self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings):
        """
        Adds new embeddings to FAISS index.
        """
        embeddings = np.array(embeddings, dtype=np.float32)
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=5):
        """
        Searches for the top_k most similar documents.
        """
        query_embedding = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_embedding, top_k)
        return indices[0]

    def save_index(self, index_path):
        """
        Saves FAISS index.
        """
        faiss.write_index(self.index, index_path)
        print(f"✅ FAISS index saved to {index_path}")
