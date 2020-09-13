from spacy_crfsuite.features import Featurizer
from spacy_crfsuite.tokenizer import SpacyTokenizer


def test_dense_features_with_spacy_sm(en_core_web_sm):
    message = {"text": "hello world"}
    tokenizer = SpacyTokenizer(en_core_web_sm)
    tokenizer.tokenize(message)

    featurizer = Featurizer(use_dense_features=True)
    dense_features = featurizer.get_dense_features(message)
    assert dense_features is None


def test_dense_features_with_spacy_md_and_flag_disabled(en_core_web_md):
    message = {"text": "hello world"}
    tokenizer = SpacyTokenizer(en_core_web_md)
    tokenizer.tokenize(message)

    featurizer = Featurizer(use_dense_features=False)
    dense_features = featurizer.get_dense_features(message)
    assert dense_features is None


def test_dense_features_with_spacy_md(en_core_web_md):
    message = {"text": "hello world"}
    tokenizer = SpacyTokenizer(en_core_web_md)
    tokenizer.tokenize(message)

    featurizer = Featurizer(use_dense_features=True)
    dense_features = featurizer.get_dense_features(message)

    assert len(dense_features) == 3
    assert len(dense_features[0]["text_dense_features"]) == 300
