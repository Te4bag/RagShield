import spacy

# Load a lightweight model for fast sentence splitting
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

# Fragments this short are dropped: stray list markers and punctuation that
# spaCy splits off as their own "sentence" and that no premise could entail.
MIN_SENTENCE_CHARS = 5


def split_into_sentence_spans(text):
    """Sentences with their character offsets, as `(start, end, sentence)`.

    `text[start:end] == sentence` always holds. The offsets exist for the eval
    loaders, which map annotated character spans onto the sentences the auditor
    will actually score; deriving `split_into_sentences` from this function is
    what guarantees the two can never segment differently.
    """
    spans = []
    for sent in nlp(text).sents:
        stripped = sent.text.strip()
        if len(stripped) > MIN_SENTENCE_CHARS:
            start = sent.start_char + (len(sent.text) - len(sent.text.lstrip()))
            spans.append((start, start + len(stripped), stripped))
    return spans


def split_into_sentences(text):
    """
    Splits a paragraph into a list of individual sentences using spaCy.
    """
    return [sentence for _, _, sentence in split_into_sentence_spans(text)]
