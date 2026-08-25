from index import RagShieldIndex, cfg

class Retriever:
    def __init__(self, index=None):
        # An index may be injected so callers can share a single instance --
        # and therefore a single loaded embedding model and a single live
        # collection handle. Without that, a rebuilt collection would leave
        # this retriever pointing at the deleted one.
        self.top_k = cfg['retrieval']['top_k']
        self.index = index if index is not None else RagShieldIndex()

    def get_context(self, query):
        results = self.index.query(query, n_results=self.top_k)
        
        # Combine documents into a single string
        context_text = "\n\n".join(results['documents'][0])
        
        # Keep metadata for attribution
        metadatas = results['metadatas'][0]
        
        return context_text, metadatas