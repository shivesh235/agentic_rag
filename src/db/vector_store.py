import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import uuid
from typing import Dict, List, Tuple
import pickle
import os

class FAISSVectorStore:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.doc_ids: List[str] = []
        
    def add_document(self, content: str) -> str:
        """Add a document to the vector store"""
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Create embedding
        embedding = self.model.encode([content])[0]
        
        # Add to FAISS index
        # Add to FAISS index with document ID as integer
        embedding_array = np.array([embedding], dtype=np.float32)
        doc_id_int = np.array([len(self.doc_ids)], dtype=np.int64)
        self.index.add(embedding)
        self.doc_ids.append(doc_id)
        
        return doc_id
    
    def search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents"""
        # Create query embedding
        query_vector = self.model.encode([query])[0]
        
        # Search in FAISS
        distances, indices = self.index.search(
            np.array([query_vector], dtype=np.float32), 
            k
        )
        
        # Return results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.doc_ids):  # Ensure valid index
                results.append((self.doc_ids[idx], float(dist)))
        
        return results
    
    def save(self, directory: str):
        """Save the vector store to disk"""
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(directory, "faiss.index"))
        
        # Save document IDs
        with open(os.path.join(directory, "doc_ids.pkl"), "wb") as f:
            pickle.dump(self.doc_ids, f)
    
    def load(self, directory: str):
        """Load the vector store from disk"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(directory, "faiss.index"))
        
        # Load document IDs
        with open(os.path.join(directory, "doc_ids.pkl"), "rb") as f:
            self.doc_ids = pickle.load(f)
