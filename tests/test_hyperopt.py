import pytest

from spacy_crfsuite import CRFExtractor, read_file
from spacy_crfsuite.tokenizer import SpacyTokenizer
from spacy_crfsuite.train import gold_example_to_crf_tokens


@pytest.fixture()
def dev_examples(en_core_web_md):
    tokenizer = SpacyTokenizer(en_core_web_md)

    dev_examples = [
        gold_example_to_crf_tokens(
            ex, tokenizer=tokenizer, use_dense_features=False, bilou=True
        ) for ex in read_file("examples/restaurent_search.md")
    ]

    return dev_examples


def test_hyperparam_optim(dev_examples):
    crf_extractor = CRFExtractor(component_config={
        "features": [
            [
                "low",
                "title",
                "upper",
                "pos",
                "pos2"
            ],
            [
                "low",
                "bias",
                "prefix5",
                "prefix2",
                "suffix5",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
                "pos",
                "pos2"
            ],
            [
                "low",
                "title",
                "upper",
                "pos",
                "pos2"
            ],
        ],
        "c1": 0.01,
        "c2": 0.22
    })

    rs = crf_extractor.fine_tune(dev_examples, cv=5, n_iter=30, random_state=42)
    assert "c1" in rs.best_params_
    assert "c2" in rs.best_params_
    assert isinstance(rs.best_score_, float)
