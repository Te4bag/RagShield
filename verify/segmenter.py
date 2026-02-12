import spacy

# Load a lightweight model for fast sentence splitting
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

def split_into_sentences(text):
    """
    Splits a paragraph into a list of individual sentences using spaCy.
    """
    doc = nlp(text)
    # Extract clean text for each sentence detected
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]