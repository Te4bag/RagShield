import yaml
import os

def load_config(config_path="config.yaml"):
    # Path relative to the root of the project
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(root_dir, config_path)
    
    if not os.path.exists(full_path):
        return {
            "ingestion": {"chunk_size": 600, "chunk_overlap": 60},
            "retrieval": {"top_k": 3},
            "models": {"embeddings": "all-MiniLM-L6-v2", "generator": "llama3-8b-8192"}
        }
    
    with open(full_path, "r") as f:
        return yaml.safe_load(f)

cfg = load_config()