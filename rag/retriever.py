from index import RagShieldIndex, cfg

class Retriever:
    def __init__(self):
        # Pull k from config. Default to 3 if missing.
        self.top_k = cfg.get('retrieval', {}).get('top_k', 3)
        self.index = RagShieldIndex()

    def get_context(self, query):
        results = self.index.query(query, n_results=self.top_k)
        
        # Combine documents into a single string
        context_text = "\n\n".join(results['documents'][0])
        
        # Keep metadata for attribution
        metadatas = results['metadatas'][0]
        
        return context_text, metadatas