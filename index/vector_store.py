import chromadb
from chromadb.utils import embedding_functions
import os
from .config_loader import cfg

# The single ChromaDB collection backing the app. Anything that creates or
# deletes it must reference this name rather than repeating the literal.
COLLECTION_NAME = "rag_shield_docs"


class RagShieldIndex:
    def __init__(self, db_path="./chroma_db"):
        # Pull model from YAML
        model_name = cfg['models']['embeddings']
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

    def add_documents(self, chunks):
        """Adds chunks to the vector store with metadata."""
        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        # chunk_id and source_type ride along so a retrieved result can be
        # attributed to the passage it came from, not merely to the document.
        # Chroma returns ids separately, but carrying chunk_id in the metadata
        # keeps attribution intact for callers that only read metadatas.
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "source_type": c["source_type"],
            }
            for c in chunks
        ]
        
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