import chromadb
from chromadb.utils import embedding_functions
import os

class RagShieldIndex:
    def __init__(self, db_path="./chroma_db"):
        # Use a local embedding model (all-MiniLM-L6-v2 is fast and reliable)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="rag_shield_docs",
            embedding_function=self.embedding_fn
        )

    def add_documents(self, chunks):
        """Adds chunks to the vector store with metadata."""
        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [{"doc_id": c["doc_id"]} for c in chunks]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        print(f" Indexed {len(chunks)} chunks into ChromaDB.")

    def query(self, text, n_results=3):
        """Search for the most relevant chunks."""
        return self.collection.query(
            query_texts=[text],
            n_results=n_results
        )