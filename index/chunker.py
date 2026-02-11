from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config_loader import cfg # Relative import

def chunk_documents(documents):
    # Pull settings from YAML
    size = cfg['ingestion']['chunk_size']
    overlap = cfg['ingestion']['chunk_overlap']
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
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