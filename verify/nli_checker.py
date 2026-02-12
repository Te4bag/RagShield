from sentence_transformers import CrossEncoder
from index import cfg
from .segmenter import split_into_sentences

class NLIAuditor:
    def __init__(self):
        # Model defined in your config.yaml (e.g., cross-encoder/nli-deberta-v3-small)
        model_name = cfg.get('models', {}).get('nli_model', 'cross-encoder/nli-deberta-v3-small')
        self.model = CrossEncoder(model_name)
        self.threshold = cfg.get('verification', {}).get('entailment_threshold', 0.65)

    def audit_response(self, response_text, retrieved_context):
        sentences = split_into_sentences(response_text)
        audit_results = []

        for sentence in sentences:
            # We compare the sentence against the entire retrieved context block
            # Cross-Encoders take a list of pairs: [(premise, hypothesis)]
            score = self.model.predict([(retrieved_context, sentence)])[0]
            
            # DeBERTa NLI outputs usually map to: 0: Contradiction, 1: Neutral, 2: Entailment
            # We convert these to a readable verdict
            label_mapping = ['CONTRADICTION', 'NEUTRAL', 'ENTAILMENT']
            verdict = label_mapping[score.argmax()]
            confidence = float(score.max())

            # Refine verdict based on your custom threshold
            if verdict == 'ENTAILMENT' and confidence < self.threshold:
                verdict = 'NEUTRAL'

            audit_results.append({
                "sentence": sentence,
                "verdict": verdict,
                "confidence": round(confidence, 2)
            })
            
        return audit_results