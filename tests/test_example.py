import requests
import spacy
import pytest
import os
import tqdm

from spacy.language import Language
from spacy_crfsuite import CRFEntityExtractor, CRFExtractor

MODEL_URL = "https://github.com/talmago/spacy_crfsuite/releases/download/v1.1.0/spacy_crfsuite_conll03_sm.bz2"


@pytest.fixture()
def spacy_crfsuite_conll03_sm():
    filename = os.path.basename(MODEL_URL)

    if not os.path.exists(filename):
        r = requests.get(MODEL_URL, stream=True)
        with open(filename, 'wb') as f:
            file_size = int(r.headers["content-length"])
            chunk_size = 1000
            with tqdm.tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)

    return filename


def test_example(spacy_crfsuite_conll03_sm):
    @Language.factory("ner_crf")
    def create_component(nlp, name):
        crf_extractor = CRFExtractor().from_disk(spacy_crfsuite_conll03_sm)
        return CRFEntityExtractor(nlp, crf_extractor=crf_extractor)

    nlp = spacy.load("en_core_web_sm", disable=["ner"])
    nlp.add_pipe("ner_crf")

    doc = nlp(
        "George Walker Bush (born July 6, 1946) is an American politician and businessman "
        "who served as the 43rd president of the United States from 2001 to 2009.")

    for ent in doc.ents:
        print(ent, "-", ent.label_)

    # Output:
    # George Walker Bush - PER
    # American - MISC
    # United States - LOC
