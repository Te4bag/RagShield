"""Aggregation logic for NLIAuditor, exercised against a stubbed cross-encoder.

No weights are loaded and nothing touches the network: `CrossEncoder` is
replaced with a stub that returns scripted logits, so the real `__init__` and
`audit_response` run end to end in milliseconds.

The sentence splitter is stubbed too. spaCy's segmentation is not what these
tests are about, and letting it drift would make aggregation tests fail for
unrelated reasons. Segmenter behaviour belongs to the loader/chunker suite.
"""
import numpy as np
import pytest

from verify import nli_checker as nli

DEBERTA_LABELS = {0: "contradiction", 1: "entailment", 2: "neutral"}
ROBERTA_LABELS = {0: "contradiction", 1: "neutral", 2: "entailment"}

# Logit rows, written in the *deberta* order (contradiction, entailment, neutral).
STRONG_ENTAIL = (0.0, 8.0, 0.0)
WEAK_ENTAIL = (0.0, 1.4, 0.0)        # softmaxes to ~0.66 -- below the 0.85 gate
STRONG_CONTRA = (8.0, 0.0, 0.0)
WEAK_CONTRA = (1.4, 0.0, 0.0)
STRONG_NEUTRAL = (0.0, 0.0, 8.0)


def softmax(logits):
    """Mirrors the implementation, so expected confidences are bit-identical."""
    row = np.asarray(logits, dtype=float)
    exp = np.exp(row - row.max())
    return exp / exp.sum()


def chunk(cid, text=None):
    return {"chunk_id": cid, "doc_id": f"{cid.split('_')[0]}.pdf",
            "text": text or f"body of {cid}"}


def build(monkeypatch, script=None, default=STRONG_NEUTRAL,
          id2label=None, sentences=None, **overrides):
    """An NLIAuditor whose model is a stub returning scripted logits.

    `script` maps (premise_text, sentence) -> logit row. Anything unscripted
    falls back to `default`, so a test only writes down the pairs it cares about.
    """
    captured = {}

    class StubCrossEncoder:
        def __init__(self, model_name=None):
            self.model_name = model_name
            self.model = type("M", (), {})()
            self.model.config = type("C", (), {})()
            self.model.config.id2label = dict(id2label or DEBERTA_LABELS)
            self.calls = []
            captured["model"] = self

        def predict(self, pairs):
            pairs = list(pairs)
            self.calls.append(pairs)
            rows = [(script or {}).get((p, h), default) for p, h in pairs]
            return np.asarray(rows, dtype=float)

    monkeypatch.setattr(nli, "CrossEncoder", StubCrossEncoder)
    for key, value in overrides.items():
        monkeypatch.setitem(nli.cfg["verification"], key, value)
    if sentences is not None:
        monkeypatch.setattr(nli, "split_into_sentences", lambda text: list(sentences))

    auditor = nli.NLIAuditor()
    auditor.stub = captured["model"]
    return auditor


# --------------------------------------------------------------- selection

def test_picks_the_chunk_with_the_highest_entailment(monkeypatch):
    """The winning chunk is the one that best *supports* the sentence."""
    s = "the encoder has six layers"
    chunks = [chunk("a_ch0"), chunk("b_ch1"), chunk("c_ch2")]
    auditor = build(monkeypatch, sentences=[s], script={
        ("body of a_ch0", s): STRONG_NEUTRAL,
        ("body of b_ch1", s): STRONG_ENTAIL,
        ("body of c_ch2", s): STRONG_NEUTRAL,
    })

    (result,) = auditor.audit_response("ignored", chunks)

    assert result["verdict"] == "ENTAILMENT"
    assert result["evidence"]["chunk_id"] == "b_ch1"
    assert result["evidence"]["doc_id"] == "b.pdf"
    assert result["evidence"]["text"] == "body of b_ch1"


def test_selection_is_entailment_not_most_confident_non_neutral(monkeypatch):
    """Regression: 'strongest non-neutral' selection favours CONTRADICTION.

    Chunk A screams contradiction, chunk B moderately entails. The rule under
    test must return B. Choosing the most confident non-neutral chunk instead
    returns A, which is how that rule manufactures false red flags.
    """
    s = "a sentence the source supports"
    chunks = [chunk("a_ch0"), chunk("b_ch1")]
    auditor = build(monkeypatch, sentences=[s], script={
        ("body of a_ch0", s): STRONG_CONTRA,          # 0.9997 contradiction
        ("body of b_ch1", s): (0.0, 3.0, 0.0),        # ~0.90 entailment
    })

    (result,) = auditor.audit_response("ignored", chunks)

    assert result["verdict"] == "ENTAILMENT"
    assert result["evidence"]["chunk_id"] == "b_ch1"


def test_contradiction_wins_when_no_chunk_entails(monkeypatch):
    """Max-entailment still surfaces a contradiction when that is the best read."""
    s = "the decoder has twelve layers"
    chunks = [chunk("a_ch0"), chunk("b_ch1")]
    auditor = build(monkeypatch, sentences=[s], script={
        ("body of a_ch0", s): STRONG_NEUTRAL,
        ("body of b_ch1", s): STRONG_CONTRA,
    })

    (result,) = auditor.audit_response("ignored", chunks)

    assert result["verdict"] == "CONTRADICTION"
    assert result["evidence"]["chunk_id"] == "b_ch1"


# --------------------------------------------------------------- thresholds

def test_low_confidence_entailment_is_demoted(monkeypatch):
    s = "weakly supported"
    auditor = build(monkeypatch, sentences=[s],
                    script={("body of a_ch0", s): WEAK_ENTAIL})

    (result,) = auditor.audit_response("ignored", [chunk("a_ch0")])

    assert result["verdict"] == "NEUTRAL"
    # The confidence still reports the winning class, not the demoted label.
    assert result["confidence"] == pytest.approx(softmax(WEAK_ENTAIL)[1], abs=0.005)


def test_entailment_exactly_at_the_threshold_survives(monkeypatch):
    """The comparison is strict `<`, so equality must not demote."""
    s = "exactly at the gate"
    exact = float(softmax(STRONG_ENTAIL)[1])
    auditor = build(monkeypatch, sentences=[s], entailment_threshold=exact,
                    script={("body of a_ch0", s): STRONG_ENTAIL})

    (result,) = auditor.audit_response("ignored", [chunk("a_ch0")])

    assert result["verdict"] == "ENTAILMENT"


def test_low_confidence_contradiction_is_demoted(monkeypatch):
    """A red underline claims the source refutes the sentence; a coin flip cannot."""
    s = "weakly contradicted"
    auditor = build(monkeypatch, sentences=[s],
                    script={("body of a_ch0", s): WEAK_CONTRA})

    (result,) = auditor.audit_response("ignored", [chunk("a_ch0")])

    assert result["verdict"] == "NEUTRAL"


def test_contradiction_exactly_at_the_floor_survives(monkeypatch):
    s = "exactly at the floor"
    exact = float(softmax(STRONG_CONTRA)[0])
    auditor = build(monkeypatch, sentences=[s], contradiction_threshold=exact,
                    script={("body of a_ch0", s): STRONG_CONTRA})

    (result,) = auditor.audit_response("ignored", [chunk("a_ch0")])

    assert result["verdict"] == "CONTRADICTION"


def test_zero_floor_disables_the_contradiction_demotion(monkeypatch):
    """`contradiction_threshold: 0.0` is the documented pre-D1 ablation."""
    s = "weakly contradicted"
    auditor = build(monkeypatch, sentences=[s], contradiction_threshold=0.0,
                    script={("body of a_ch0", s): WEAK_CONTRA})

    (result,) = auditor.audit_response("ignored", [chunk("a_ch0")])

    assert result["verdict"] == "CONTRADICTION"


def test_neutral_is_never_promoted(monkeypatch):
    """Both thresholds are one-way. Nothing turns a NEUTRAL into anything else."""
    s = "unsupported but not refuted"
    auditor = build(monkeypatch, sentences=[s], entailment_threshold=0.0,
                    contradiction_threshold=0.0,
                    script={("body of a_ch0", s): STRONG_NEUTRAL})

    (result,) = auditor.audit_response("ignored", [chunk("a_ch0")])

    assert result["verdict"] == "NEUTRAL"


# --------------------------------------------------------------- premises

def test_each_sentence_is_scored_against_every_chunk_separately(monkeypatch):
    sentences = ["first sentence", "second sentence"]
    chunks = [chunk("a_ch0"), chunk("b_ch1"), chunk("c_ch2")]
    auditor = build(monkeypatch, sentences=sentences)

    auditor.audit_response("ignored", chunks)

    # One batched predict() per sentence, three pairs in each.
    assert len(auditor.stub.calls) == len(sentences)
    for call in auditor.stub.calls:
        assert len(call) == len(chunks)
        assert [p for p, _ in call] == [c["text"] for c in chunks]


def test_concatenate_mode_collapses_to_a_single_premise(monkeypatch):
    """The eval baseline joins every chunk into one premise, blank-line separated."""
    sentences = ["only sentence"]
    chunks = [chunk("a_ch0"), chunk("b_ch1")]
    auditor = build(monkeypatch, sentences=sentences, aggregation="concatenate")

    auditor.audit_response("ignored", chunks)

    (call,) = auditor.stub.calls
    assert len(call) == 1
    assert call[0][0] == "body of a_ch0\n\nbody of b_ch1"


def test_string_context_recovers_chunk_boundaries(monkeypatch):
    """A joined string still gets split back into per-chunk premises."""
    sentences = ["only sentence"]
    auditor = build(monkeypatch, sentences=sentences)

    auditor.audit_response("ignored", "chunk one\n\nchunk two\n\nchunk three")

    (call,) = auditor.stub.calls
    assert [p for p, _ in call] == ["chunk one", "chunk two", "chunk three"]


def test_empty_context_returns_neutral_rows_without_dropping_sentences(monkeypatch):
    sentences = ["one", "two", "three"]
    auditor = build(monkeypatch, sentences=sentences)

    results = auditor.audit_response("ignored", [])

    assert len(results) == len(sentences)
    assert [r["verdict"] for r in results] == ["NEUTRAL"] * 3
    assert all(r["evidence"] is None for r in results)
    assert auditor.stub.calls == []          # the model is never consulted


def test_blank_chunks_are_skipped(monkeypatch):
    sentences = ["only sentence"]
    chunks = [chunk("a_ch0", "   "), chunk("b_ch1", "real text")]
    auditor = build(monkeypatch, sentences=sentences)

    auditor.audit_response("ignored", chunks)

    (call,) = auditor.stub.calls
    assert [p for p, _ in call] == ["real text"]


# --------------------------------------------------------------- construction

def test_label_order_is_read_from_the_checkpoint(monkeypatch):
    """The bug that inverted the product: deberta and roberta disagree."""
    deberta = build(monkeypatch, id2label=DEBERTA_LABELS)
    assert deberta.label_order == ["CONTRADICTION", "ENTAILMENT", "NEUTRAL"]
    assert deberta.entailment_index == 1


def test_roberta_ordering_is_handled(monkeypatch):
    roberta = build(monkeypatch, id2label=ROBERTA_LABELS)
    assert roberta.label_order == ["CONTRADICTION", "NEUTRAL", "ENTAILMENT"]
    assert roberta.entailment_index == 2


def test_roberta_ordering_changes_which_logit_is_entailment(monkeypatch):
    """Same logits, different checkpoint ordering, correctly different verdict.

    Under the roberta ordering index 2 is entailment, so a row peaking at index
    1 is NEUTRAL -- the exact case the hardcoded list got backwards.
    """
    s = "a sentence"
    auditor = build(monkeypatch, sentences=[s], id2label=ROBERTA_LABELS,
                    script={("body of a_ch0", s): (0.0, 8.0, 0.0)})

    (result,) = auditor.audit_response("ignored", [chunk("a_ch0")])

    assert result["verdict"] == "NEUTRAL"


def test_checkpoint_without_nli_labels_raises(monkeypatch):
    with pytest.raises(ValueError, match="three NLI labels"):
        build(monkeypatch, id2label={0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})


def test_unknown_aggregation_raises(monkeypatch):
    with pytest.raises(ValueError, match="Unknown verification.aggregation"):
        build(monkeypatch, aggregation="nonsense")
