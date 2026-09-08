import hashlib

import chromadb
from chromadb.utils import embedding_functions

from .chunker import chunk_documents
from .config_loader import cfg

# The single ChromaDB collection backing the app. Anything that creates or
# deletes it must reference this name rather than repeating the literal.
COLLECTION_NAME = "rag_shield_docs"


def document_hash(text):
    """Content hash of a document's extracted text.

    Changes if and only if the text changes, so it is what decides whether a
    document needs re-embedding. Truncated because it is an equality check, not
    a security boundary.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def plan_sync(existing, incoming):
    """Work out what to embed, replace and drop.

    `existing` and `incoming` are both {doc_id: content_hash}. Kept as a pure
    function so the decision can be tested without a database.

    A document already in the collection with a matching hash is left alone --
    that is the whole point, since re-embedding an unchanged corpus is the cost
    that made adding one file to a large one so slow.
    """
    added = sorted(d for d in incoming if d not in existing)
    updated = sorted(d for d in incoming
                     if d in existing and existing[d] != incoming[d])
    unchanged = sorted(d for d in incoming
                       if d in existing and existing[d] == incoming[d])
    removed = sorted(d for d in existing if d not in incoming)
    return {"added": added, "updated": updated,
            "unchanged": unchanged, "removed": removed}


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
        # doc_hash is what lets sync_documents skip an unchanged document.
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "source_type": c["source_type"],
                "doc_hash": c.get("doc_hash", ""),
            }
            for c in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
        print(f" Indexed {len(chunks)} chunks into ChromaDB.")

    def indexed_hashes(self):
        """{doc_id: content_hash} for everything currently in the collection.

        A collection written before doc_hash existed reports an empty hash for
        those documents, which never matches an incoming one -- so they are
        re-indexed once and are correct from then on.
        """
        stored = self.collection.get(include=["metadatas"])
        hashes = {}
        for meta in (stored.get("metadatas") or []):
            if meta and meta.get("doc_id"):
                hashes[meta["doc_id"]] = meta.get("doc_hash", "")
        return hashes

    def remove_document(self, doc_id):
        """Drop every chunk belonging to one document."""
        self.collection.delete(where={"doc_id": doc_id})

    def sync_documents(self, documents):
        """Make the collection match `documents`, embedding only what changed.

        Replaces the delete-everything-and-rebuild path: adding one file to a
        corpus previously re-embedded the entire corpus. Returns the plan that
        was applied, so the caller can report what actually happened.
        """
        incoming = {d["doc_id"]: document_hash(d["text"]) for d in documents}
        plan = plan_sync(self.indexed_hashes(), incoming)

        # A changed document is dropped and re-added rather than updated in
        # place: its text may now chunk into a different number of pieces, so
        # the old chunk ids are not a subset of the new ones.
        for doc_id in plan["updated"] + plan["removed"]:
            self.remove_document(doc_id)

        stale = set(plan["added"]) | set(plan["updated"])
        if stale:
            chunks = chunk_documents([d for d in documents if d["doc_id"] in stale])
            for chunk in chunks:
                chunk["doc_hash"] = incoming[chunk["doc_id"]]
            self.add_documents(chunks)

        return plan

    def query(self, text, n_results=3):
        """Search for the most relevant chunks."""
        return self.collection.query(
            query_texts=[text],
            n_results=n_results
        )
