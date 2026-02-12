from sentence_transformers import CrossEncoder
from index import cfg
from .segmenter import split_into_sentences
import numpy as np

class NLIAuditor:
    def __init__(self):
        # Model defined in config.yaml
        model_name = cfg.get('models', {}).get('nli_model', 'cross-encoder/nli-deberta-v3-small')
        self.model = CrossEncoder(model_name)
        self.threshold = cfg.get('verification', {}).get('entailment_threshold', 0.65)

    def audit_response(self, response_text, retrieved_context):
        sentences = split_into_sentences(response_text)
        audit_results = []

        for sentence in sentences:
            # 1. Get raw logits from the model
            logits = self.model.predict([(retrieved_context, sentence)])[0]
            
            # 2. Apply Softmax to convert logits to probabilities (0.0 to 1.0)
            exp_logits = np.exp(logits)
            probs = exp_logits / exp_logits.sum()
            
            # 3. Map probabilities to labels
            # 0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT
            label_mapping = ['CONTRADICTION', 'NEUTRAL', 'ENTAILMENT']
            verdict_idx = np.argmax(probs)
            verdict = label_mapping[verdict_idx]
            confidence = float(probs[verdict_idx])

            # 4. Refine verdict based on our custom threshold
            # If the model says ENTAILMENT but is not confident, mark as NEUTRAL
            if verdict == 'ENTAILMENT' and confidence < self.threshold:
                verdict = 'NEUTRAL'
            
            # 5. Handle "Double Negative" Logic
            # If the LLM says "I don't know" and the PDF doesn't mention it, the model correctly flags that statement as ENTAILMENT.

            audit_results.append({
                "sentence": sentence,
                "verdict": verdict,
                "confidence": round(confidence, 2)
            })
            
        return audit_results