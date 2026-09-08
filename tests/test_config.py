"""Config loading: the deep merge, and that DEFAULTS matches config.yaml.

The merge is what makes a partial `config.yaml` safe. Before it, the loader
returned an all-or-nothing fallback dict, so a config file that omitted a key
silently lost every other key in that section — and the fallback had drifted
from the YAML, so the value you got depended on whether the file loaded.
"""
import copy

import yaml

from index.config_loader import DEFAULTS, _deep_merge, load_config

SECTIONS = {"ingestion", "retrieval", "models", "verification"}


# ---------------------------------------------------------------- deep merge

def test_nested_override_keeps_its_siblings():
    """The bug this exists to prevent: overriding one key wiping the rest."""
    base = {"verification": {"entailment_threshold": 0.85, "batch_size": 32}}

    merged = _deep_merge(base, {"verification": {"batch_size": 8}})

    assert merged["verification"] == {"entailment_threshold": 0.85, "batch_size": 8}


def test_untouched_sections_survive():
    merged = _deep_merge(DEFAULTS, {"retrieval": {"top_k": 9}})

    assert merged["retrieval"]["top_k"] == 9
    assert merged["models"] == DEFAULTS["models"]
    assert merged["verification"] == DEFAULTS["verification"]


def test_scalar_replaces_scalar():
    assert _deep_merge({"a": 1}, {"a": 2}) == {"a": 2}


def test_non_dict_override_replaces_a_dict():
    """A YAML author who writes `verification: null` gets exactly that."""
    assert _deep_merge({"a": {"b": 1}}, {"a": None}) == {"a": None}


def test_keys_absent_from_the_base_are_added():
    assert _deep_merge({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_none_override_is_treated_as_empty():
    """An empty config.yaml parses to None; it must not blow up."""
    assert _deep_merge(DEFAULTS, None) == DEFAULTS


def test_merge_does_not_mutate_the_base():
    """DEFAULTS is module-level state shared by every caller."""
    before = copy.deepcopy(DEFAULTS)

    _deep_merge(DEFAULTS, {"verification": {"batch_size": 1}, "retrieval": {"top_k": 99}})

    assert DEFAULTS == before


def test_merge_result_is_deeply_independent_of_the_base():
    merged = _deep_merge(DEFAULTS, {})

    merged["verification"]["batch_size"] = 12345

    assert DEFAULTS["verification"]["batch_size"] != 12345


# --------------------------------------------------------------- load_config

def test_missing_file_falls_back_to_defaults():
    cfg = load_config("definitely-not-a-real-config.yaml")

    assert cfg == DEFAULTS


def test_missing_file_returns_a_copy_not_the_original():
    cfg = load_config("definitely-not-a-real-config.yaml")

    cfg["retrieval"]["top_k"] = 4242

    assert DEFAULTS["retrieval"]["top_k"] != 4242


def test_every_section_is_present_after_loading():
    cfg = load_config()

    assert SECTIONS <= set(cfg)


def test_call_sites_can_subscript_directly():
    """Call sites use cfg['a']['b'] with no fallback, so keys must exist."""
    cfg = load_config()

    cfg["ingestion"]["chunk_size"]
    cfg["ingestion"]["chunk_overlap"]
    cfg["retrieval"]["top_k"]
    cfg["models"]["embeddings"]
    cfg["models"]["generator"]
    cfg["models"]["nli_model"]
    cfg["verification"]["entailment_threshold"]
    cfg["verification"]["contradiction_threshold"]
    cfg["verification"]["aggregation"]
    cfg["verification"]["batch_size"]
    cfg["verification"]["max_length"]


# ------------------------------------------------- DEFAULTS vs config.yaml

def _yaml_config():
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_yaml_declares_no_key_that_defaults_does_not():
    """A YAML-only key resolves fine but is invisible when the file is missing.

    That is exactly the drift the single-DEFAULTS change was meant to end, so
    it is worth failing the build over.
    """
    y = _yaml_config()
    unknown = {(s, k) for s, block in y.items() if isinstance(block, dict)
               for k in block if k not in DEFAULTS.get(s, {})}

    assert not unknown, f"in config.yaml but not DEFAULTS: {sorted(unknown)}"


def test_shared_keys_hold_the_same_values():
    """CLAUDE.md: 'Keep DEFAULTS and config.yaml in sync.' Enforced here.

    They are meant to describe the same configuration, so a silent divergence
    means the behaviour depends on whether config.yaml was found.
    """
    y = _yaml_config()
    mismatched = {
        (s, k): (v, DEFAULTS[s][k])
        for s, block in y.items() if isinstance(block, dict)
        for k, v in block.items()
        if k in DEFAULTS.get(s, {}) and DEFAULTS[s][k] != v
    }

    assert not mismatched, f"config.yaml vs DEFAULTS disagree: {mismatched}"
