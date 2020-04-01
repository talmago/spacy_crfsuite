import pytest

from spacy_crfsuite.dataset import read_markdown


@pytest.fixture()
def markdown_fmt():
    return """## intent
    - what is my balance <!-- no entity -->
    - how much do I have on my [savings](source_account) <!-- entity "source_account" has value "savings" -->
    - how much do I have on my [savings account](source_account:savings) <!-- synonyms, method 1-->
    - Could I pay in [yen](currency)?  <!-- entity matched by lookup table -->"""


def test_read_markdown(markdown_fmt):
    assert read_markdown(markdown_fmt) == [
        {'entities': [], 'text': 'what is my balance'},
        {'entities': [{'end': 32,
                       'entity': 'source_account',
                       'start': 25,
                       'value': 'savings'}],
         'text': 'how much do I have on my savings'},
        {'entities': [{'end': 40,
                       'entity': 'source_account',
                       'start': 25,
                       'value': 'savings'}],
         'text': 'how much do I have on my savings account'},
        {'entities': [{'end': 18,
                       'entity': 'currency',
                       'start': 15,
                       'value': 'yen'}],
         'text': 'Could I pay in yen?'}
    ]
