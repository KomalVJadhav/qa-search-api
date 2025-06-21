# src/query_classifier.py

import stanza

class QueryClassifier:
    def __init__(self):
        stanza.download("en", verbose=False)
        self.nlp = stanza.Pipeline(lang="en", processors="tokenize,pos", use_gpu=False)

    def classify(self, query: str) -> str:
        doc = self.nlp(query)
        pos_tags = [word.xpos for sent in doc.sentences for word in sent.words]

        # Simple rule-based classification
        if query.endswith("?") or "WP" in pos_tags or "VB" in pos_tags:
            return "question"
        elif any(tag.startswith("NN") for tag in pos_tags):
            return "statement"
        else:
            return "keyword"
