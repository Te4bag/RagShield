import warnings

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
        # Stated rather than inherited: the truncation guard below compares
        # against this number, so it must be the one the model actually uses.
        self.max_length = cfg['verification']['max_length']
        self.model = CrossEncoder(model_name, max_length=self.max_length)
        self.threshold = cfg['verification']['entailment_threshold']
        self.contradiction_threshold = cfg['verification']['contradiction_threshold']
        self.aggregation = cfg['verification']['aggregation']
        self.batch_size = cfg['verification']['batch_size']
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"Unknown verification.aggregation {self.aggregation!r}; "
                f"expected one of {AGGREGATIONS}."
            )
        self.label_order = _label_order(self.model, model_name)
        self.entailment_index = self.label_order.index('ENTAILMENT')
        self._truncation_warned = False

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

    def _warn_if_truncated(self, pairs):
        """Warn once if any (premise, sentence) pair exceeds the input budget.

        The tokenizer truncates silently. A sentence judged against a truncated
        premise is judged against a *partial* source, and in the output that is
        indistinguishable from a sentence the source genuinely fails to support
        -- so the failure is invisible exactly where it matters.

        Not a live problem at `top_k: 3` with per-chunk aggregation, where the
        longest pair measured 134 tokens against 512. This guards the settings
        that would make it one: a larger `top_k`, a larger `chunk_size`, denser
        source text, or the `concatenate` baseline, whose premise is every
        retrieved chunk joined together.
        """
        if self._truncation_warned or not pairs:
            return
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return

        encoded = tokenizer([p for p, _ in pairs], [h for _, h in pairs],
                            truncation=False)
        longest = max(len(ids) for ids in encoded["input_ids"])
        if longest <= self.max_length:
            return

        # Once per auditor: the same over-long corpus would otherwise warn on
        # every query for the life of the process.
        self._truncation_warned = True
        warnings.warn(
            f"NLI input truncated: the longest (premise, sentence) pair is "
            f"{longest} tokens against a {self.max_length}-token budget, so "
            f"{sum(len(i) > self.max_length for i in encoded['input_ids'])} of "
            f"{len(pairs)} pairs lose their tail. Those sentences are being "
            f"judged against a partial source. Lower retrieval.top_k or "
            f"ingestion.chunk_size, or raise verification.max_length if the "
            f"checkpoint supports a longer input.",
            RuntimeWarning,
            stacklevel=3,
        )

    def _distributions(self, pairs):
        """Softmaxed label distributions for every (premise, sentence) pair.

        One forward pass for the whole response. CrossEncoder.predict sorts by
        length internally and pads within a batch, so grouping more pairs
        together does not change any individual pair's logits.
        """
        logits = np.asarray(
            self.model.predict(pairs, batch_size=self.batch_size), dtype=float
        )
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

        if not sentences:
            return []

        premise_texts = [p["text"] for p in premises]

        # Every (premise, sentence) pair in one batched pass, then reshaped back
        # into per-sentence blocks. Sentence-major order, so block i belongs to
        # sentence i. P3 multiplied the pair count by top_k, which is what makes
        # batching worth doing here rather than one call per sentence.
        pairs = [(text, sentence) for sentence in sentences for text in premise_texts]
        self._warn_if_truncated(pairs)
        probs = self._distributions(pairs).reshape(
            len(sentences), len(premise_texts), -1
        )

        audit_results = []

        for sentence, block in zip(sentences, probs):
            # The premise that best supports the sentence wins, and its full
            # distribution decides the verdict. Selecting on "most confident
            # non-neutral" instead would actively favour CONTRADICTION and
            # manufacture false red flags.
            best = int(np.argmax(block[:, self.entailment_index]))
            row = block[best]

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
