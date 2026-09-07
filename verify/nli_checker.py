from sentence_transformers import CrossEncoder
from index import cfg
from .segmenter import split_into_sentences
import numpy as np

# Canonical lowercase names every HuggingFace NLI checkpoint uses in its config.
_REQUIRED_LABELS = {'contradiction', 'neutral', 'entailment'}

# How a sentence is scored against the retrieved chunks.
#   max_entailment - score each chunk as its own premise and keep the chunk
#                    with the highest entailment probability (SummaC-style).
#   concatenate    - join every chunk into one premise. Retained only as an
#                    evaluation baseline; it collapses most verdicts to NEUTRAL.
AGGREGATIONS = ('max_entailment', 'concatenate')


def _label_order(model, model_name):
    """Return the verdict for each output index, read from the checkpoint.

    Checkpoints disagree on ordering: roberta-large-mnli emits
    (contradiction, neutral, entailment), while cross-encoder/nli-deberta-v3-*
    emits (contradiction, entailment, neutral). Hardcoding either order
    silently mislabels the other -- swapping entailment with neutral, which
    reports unsupported sentences as verified -- so read id2label instead of
    assuming.
    """
    id2label = model.model.config.id2label
    order = [str(id2label[key]).upper() for key in sorted(id2label, key=int)]
    if {label.lower() for label in order} != _REQUIRED_LABELS:
        raise ValueError(
            f"NLI model '{model_name}' does not expose the three NLI labels; "
            f"its id2label is {id2label!r}. RagShield needs a model whose "
            f"config names contradiction, neutral and entailment."
        )
    return order


class NLIAuditor:
    def __init__(self):
        # Defaults live in index.config_loader.DEFAULTS, so cfg is always fully
        # populated -- a missing key is a config bug and should raise here.
        model_name = cfg['models']['nli_model']
        self.model = CrossEncoder(model_name)
        self.threshold = cfg['verification']['entailment_threshold']
        self.contradiction_threshold = cfg['verification']['contradiction_threshold']
        self.aggregation = cfg['verification']['aggregation']
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Unknown verification.aggregation {self.aggregation!r}; "
                f"expected one of {AGGREGATIONS}."
            )
        self.label_order = _label_order(self.model, model_name)
        self.entailment_index = self.label_order.index('ENTAILMENT')

    def _premises(self, retrieved_context):
        """Normalise the context into a list of premise records.

        Accepts either the structured output of `Retriever.retrieve()` or the
        single joined string from `Retriever.get_context()`. In the string case
        the chunk boundaries are recovered by splitting on blank lines, which is
        exactly how get_context() joined them. Splitting more finely than the
        original chunks would be harmless anyway -- a smaller premise is what
        SummaC prescribes.
        """
        if isinstance(retrieved_context, str):
            parts = [part.strip() for part in retrieved_context.split("\n\n")]
            return [
                {"text": part, "chunk_id": None, "doc_id": None}
                for part in parts if part
            ]

        premises = []
        for chunk in retrieved_context or []:
            text = (chunk.get("text") or "").strip()
            if text:
                premises.append({
                    "text": text,
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                })
        return premises

    def _distributions(self, premise_texts, sentence):
        """Softmaxed label distribution for `sentence` against each premise."""
        pairs = [(text, sentence) for text in premise_texts]
        logits = np.asarray(self.model.predict(pairs), dtype=float)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        # Shifted by the row max for numerical stability.
        exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=1, keepdims=True)

    def audit_response(self, response_text, retrieved_context):
        sentences = split_into_sentences(response_text)
        premises = self._premises(retrieved_context)

        if self.aggregation == 'concatenate':
            premises = [{
                "text": "\n\n".join(p["text"] for p in premises),
                "chunk_id": None,
                "doc_id": None,
            }] if premises else []

        if not premises:
            # Nothing to check against. Say so rather than dropping sentences.
            return [
                {"sentence": s, "verdict": "NEUTRAL", "confidence": 0.0,
                 "evidence": None}
                for s in sentences
            ]

        premise_texts = [p["text"] for p in premises]
        audit_results = []

        for sentence in sentences:
            probs = self._distributions(premise_texts, sentence)

            # The premise that best supports the sentence wins, and its full
            # distribution decides the verdict. Selecting on "most confident
            # non-neutral" instead would actively favour CONTRADICTION and
            # manufacture false red flags.
            best = int(np.argmax(probs[:, self.entailment_index]))
            row = probs[best]

            verdict_idx = int(np.argmax(row))
            verdict = self.label_order[verdict_idx]
            confidence = float(row[verdict_idx])

            # One-way downgrades, in both directions: a verdict the model is not
            # confident about falls back to NEUTRAL. Nothing is ever promoted.
            # CONTRADICTION gets its own floor because a red underline is the
            # highest-stakes claim in the UI -- it tells a reader the source
            # actively refutes the sentence, and a coin-flip is not grounds for
            # that. Unsupported and refuted are different claims; NEUTRAL is the
            # honest verdict when the model cannot tell them apart.
            if verdict == 'ENTAILMENT' and confidence < self.threshold:
                verdict = 'NEUTRAL'
            elif verdict == 'CONTRADICTION' and confidence < self.contradiction_threshold:
                verdict = 'NEUTRAL'

            winner = premises[best]
            audit_results.append({
                "sentence": sentence,
                "verdict": verdict,
                "confidence": round(confidence, 2),
                "evidence": {
                    "chunk_id": winner["chunk_id"],
                    "doc_id": winner["doc_id"],
                    "text": winner["text"],
                },
            })

        return audit_results
