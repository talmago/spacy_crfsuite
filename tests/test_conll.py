import pytest

from spacy_crfsuite.conll import _parse_conll


@pytest.fixture()
def conll_03():
    return """The DT B-NP O
European NNP I-NP B-ORG
Commission NNP I-NP I-ORG
said VBD B-VP O
on IN B-PP O
Thursday NNP B-NP O
it PRP B-NP O
disagreed VBD B-VP O
with IN B-PP O
German JJ B-NP B-MISC
advice NN I-NP O
to TO B-PP O
consumers NNS B-NP O
to TO B-VP O
shun VB I-VP O
British JJ B-NP B-MISC
lamb NN I-NP O
until IN B-SBAR O
scientists NNS B-NP O
determine VBP B-VP O
whether IN B-SBAR O
mad JJ B-NP O
cow NN I-NP O
disease NN I-NP O
can MD B-VP O
be VB I-VP O
transmitted VBN I-VP O
to TO B-PP O
sheep NN B-NP O
. . O O"""


def test_parse_conll(conll_03):
    example = next(_parse_conll(conll_03.split("\n")))
    assert example["text"] == ("The European Commission said on Thursday it disagreed with "
                               "German advice to consumers to shun British lamb until scientists "
                               "determine whether mad cow disease can be transmitted to sheep .")

    assert example["tokens"][0].text == "The"
    assert example["tokens"][0].start == 0
    assert example["tokens"][0].get("pos") == "DT"

    assert example["tokens"][1].text == "European"
    assert example["tokens"][1].start == 4
    assert example["tokens"][1].get("pos") == "NNP"
