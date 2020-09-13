from spacy_crfsuite.tokenizer import SpacyTokenizer


def test_cls(en_core_web_sm):
    message = {"text": "hello world"}
    tokenizer = SpacyTokenizer(en_core_web_sm)
    tokenizer.tokenize(message)

    assert len(message["tokens"]) == 3
    assert message["tokens"][0].text == "hello"
    assert message["tokens"][1].text == "world"
    assert message["tokens"][2].text == "__CLS__"


def test_vectors_with_spacy_sm(en_core_web_sm):
    message = {"text": "hello world"}
    tokenizer = SpacyTokenizer(en_core_web_sm)
    tokenizer.tokenize(message)

    assert message["tokens"][0].get("vector") is None
    assert message["tokens"][1].get("vector") is None
    assert message["tokens"][2].get("vector") is None


def test_vectors_with_spacy_md(en_core_web_md):
    message = {"text": "hello world"}
    tokenizer = SpacyTokenizer(en_core_web_md)
    tokenizer.tokenize(message)

    assert message["tokens"][0].get("vector").shape == (300,)
    assert message["tokens"][1].get("vector").shape == (300,)
    assert message["tokens"][2].get("vector") is None  # CLS vector will be computed later
