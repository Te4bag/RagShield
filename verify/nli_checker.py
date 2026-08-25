from sentence_transformers import CrossEncoder
from index import cfg
from .segmenter import split_into_sentences
import numpy as np

# Canonical lowercase names every HuggingFace NLI checkpoint uses in its config.
_REQUIRED_LABELS = {'contradiction', 'neutral', 'entailment'}


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
        self.label_order = _label_order(self.model, model_name)

    def audit_response(self, response_text, retrieved_context):
        sentences = split_into_sentences(response_text)
        audit_results = []

        for sentence in sentences:
            # predict() applies no activation, so these are raw logits.
            logits = self.model.predict([(retrieved_context, sentence)])[0]

            # Softmax, shifted by the max for numerical stability.
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            verdict_idx = int(np.argmax(probs))
            verdict = self.label_order[verdict_idx]
            confidence = float(probs[verdict_idx])

            # One-way downgrade: an ENTAILMENT the model is not confident about
            # becomes NEUTRAL. Nothing is ever promoted.
            if verdict == 'ENTAILMENT' and confidence < self.threshold:
                verdict = 'NEUTRAL'

            audit_results.append({
                "sentence": sentence,
                "verdict": verdict,
                "confidence": round(confidence, 2)
            })

        return audit_results
