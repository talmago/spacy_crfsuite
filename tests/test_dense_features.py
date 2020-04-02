import pytest
import spacy

from spacy_crfsuite.dense_features import DenseFeatures


@pytest.fixture()
def nlp():
    return spacy.load('en_core_web_sm')


def test_dense_features_shape(nlp):
    dense_features = DenseFeatures(nlp)
    features = dense_features({"text": "hello world"}, attribute="text")
    assert features.shape == (3, 96)
