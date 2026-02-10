from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    # 500 characters is roughly 100-125 tokens. 
    # Good for granular NLI verification later.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=60,
        separators=["\n\n", "\n", ".", " "]
    )
    
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_ch{i}",
                "doc_id": doc["doc_id"],
                "text": text,
                "source_type": doc["source_type"]
            })
    return all_chunks