import pytest
import spacy


@pytest.fixture(scope="session", autouse=True)
def en_core_web_sm():
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="session", autouse=True)
def en_core_web_md():
    return spacy.load("en_core_web_md")
