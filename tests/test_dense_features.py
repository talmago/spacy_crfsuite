import spacy

from spacy_crfsuite.features import Featurizer
from spacy_crfsuite.tokenizer import SpacyTokenizer


def test_dense_features():
    nlp = spacy.load('en_core_web_sm')
    message = {"text": "hello world"}

    tokenizer = SpacyTokenizer(nlp)
    tokenizer.tokenize(message)

    featurizer = Featurizer(use_dense_features=True)
    dense_features = featurizer.get_dense_features(message)

    assert len(dense_features) == 3
    assert len(dense_features[0]["text_dense_features"]) == 96

    featurizer = Featurizer(use_dense_features=False)
    dense_features = featurizer.get_dense_features(message)
    assert dense_features is None
