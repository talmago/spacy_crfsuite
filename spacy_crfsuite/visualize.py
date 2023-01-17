import gradio as gr
import spacy

from spacy.language import Language
from spacy_crfsuite import CRFEntityExtractor, CRFExtractor


@Language.factory("ner_crf")
def create_component(nlp, name):
    crf_extractor = CRFExtractor().from_disk("spacy_crfsuite_conll03_sm.bz2")
    return CRFEntityExtractor(nlp, crf_extractor=crf_extractor)


nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.add_pipe("ner_crf")

examples = [
    "George Walker Bush (born July 6, 1946) is an American politician and businessman "
    "who served as the 43rd president of the United States from 2001 to 2009."
]


def ner(text):
    doc = nlp(text)
    entities = [
        {
            "entity": ent.label_,
            "score": 1.0,
            "word": ent.text,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]
    return {"text": text, "entities": entities}


demo = gr.Interface(
    ner,
    gr.Textbox(placeholder="Enter sentence here..."),
    gr.HighlightedText(),
    examples=examples,
)

demo.launch()
