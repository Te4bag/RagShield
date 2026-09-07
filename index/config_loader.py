import copy
import os

import yaml

# Single source of truth for configuration defaults.
#
# config.yaml is overlaid on top of this dict key-by-key, so a partial config
# file is valid and any key it omits resolves to the value documented here --
# not to a second, divergent set of literals. Keep this in sync with
# config.yaml; they are meant to describe the same configuration.
DEFAULTS = {
    "ingestion": {"chunk_size": 600, "chunk_overlap": 60},
    "retrieval": {"top_k": 3},
    "models": {
        "embeddings": "all-MiniLM-L6-v2",
        "generator": "llama-3.1-8b-instant",
        "nli_model": "cross-encoder/nli-deberta-v3-small",
    },
    "verification": {
        "entailment_threshold": 0.85,
        "contradiction_threshold": 0.85,
        "aggregation": "max_entailment",
    },
}


def _deep_merge(base, override):
    """Overlay `override` onto a copy of `base`, recursing into nested dicts."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path="config.yaml"):
    # Path is resolved relative to the project root, not the CWD.
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(root_dir, config_path)

    if not os.path.exists(full_path):
        return copy.deepcopy(DEFAULTS)

    with open(full_path, "r", encoding="utf-8") as f:
        return _deep_merge(DEFAULTS, yaml.safe_load(f))


cfg = load_config()
