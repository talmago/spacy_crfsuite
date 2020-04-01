import os
import spacy

from spacy_crfsuite import CRFEntityExtractorFactory

if __name__ == "__main__":

    nlp = spacy.blank('en')

    # Will look for model --> ``examples/crf.pkl``
    pipe = CRFEntityExtractorFactory(nlp, model_dir=os.getcwd())
    nlp.add_pipe(pipe)

    doc = nlp("given we launched L&M a couple of years ago")
    for ent in doc.ents:
        print(ent.text, "--", ent.label_)
