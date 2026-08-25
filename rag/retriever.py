from index import RagShieldIndex, cfg


class Retriever:
    def __init__(self, index=None):
        # An index may be injected so callers can share a single instance --
        # and therefore a single loaded embedding model and a single live
        # collection handle. Without that, a rebuilt collection would leave
        # this retriever pointing at the deleted one.
        self.top_k = cfg['retrieval']['top_k']
        self.index = index if index is not None else RagShieldIndex()

    def retrieve(self, query):
        """Return the top-k chunks as dicts, nearest first.

        Each chunk carries its own identity, so a verdict can be attributed to
        the passage that produced it rather than only to the document.
        `distance` is Chroma's similarity distance (lower is nearer), or None
        if the backend omitted it.
        """
        results = self.index.query(query, n_results=self.top_k)

        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        ids = results['ids'][0]
        # Chroma returns distances by default, but the key is not guaranteed.
        distances = (results.get('distances') or [[None] * len(documents)])[0]

        chunks = []
        for text, meta, stored_id, distance in zip(documents, metadatas, ids, distances):
            meta = meta or {}
            chunks.append({
                "text": text,
                "doc_id": meta.get("doc_id"),
                # Falls back to the stored id for collections written before
                # chunk_id was carried in the metadata.
                "chunk_id": meta.get("chunk_id", stored_id),
                "source_type": meta.get("source_type"),
                "distance": distance,
            })
        return chunks

    def get_context(self, query):
        """The top-k chunks joined into one premise string, plus metadata.

        Retained for callers that still want the concatenated context. Scoring
        a sentence against this blob is what collapses NLI verdicts to NEUTRAL,
        so the verification path should use retrieve() instead.
        """
        chunks = self.retrieve(query)
        context_text = "\n\n".join(c["text"] for c in chunks)
        metadatas = [
            {
                "doc_id": c["doc_id"],
                "chunk_id": c["chunk_id"],
                "source_type": c["source_type"],
            }
            for c in chunks
        ]
        return context_text, metadatas
