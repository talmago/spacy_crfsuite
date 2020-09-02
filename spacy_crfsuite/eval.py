import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import plac
import spacy
import srsly

from wasabi import msg

from spacy_crfsuite.crf_extractor import CRFExtractor, prepare_example
from spacy_crfsuite.dense_features import DenseFeatures
from spacy_crfsuite.tokenizer import SpacyTokenizer
from spacy_crfsuite.utils import read_file


@plac.annotations(
    in_file=("Path to input file (either .json, .md or .conll)", "positional", None, str),
    model_file=("Path to model file", "option", "m", str),
    config_file=("Path to config file (.json format)", "option", "c", str),
    spacy_model=("Name of spaCy model to use", "positional", None, str),
)
def main(in_file, model_file=None, config_file=None, spacy_model=None):
    """Train CRF entity tagger."""
    if config_file:
        msg.info(f"Loading config: {config_file}")
        component_config = srsly.read_json(config_file)
    else:
        component_config = None

    model_file = model_file or "model.pkl"
    msg.info("Loading model from file", model_file)
    crf_extractor = CRFExtractor(component_config=component_config).from_disk(model_file)
    msg.good("Successfully loaded CRF tagger", crf_extractor)

    msg.info("Loading dev dataset from file", in_file)
    dev_examples = read_file(in_file)
    msg.good(f"Successfully loaded {len(dev_examples)} dev examples.")

    if spacy_model is not None:
        nlp = spacy.load(spacy_model)
        msg.info(f"Using spaCy model: {spacy_model}")
    else:
        nlp = spacy.blank("en")
        msg.info(f"Using spaCy blank: 'en'")

    tokenizer = SpacyTokenizer(nlp=nlp)

    if crf_extractor.use_dense_features():
        dense_features = DenseFeatures(nlp)
    else:
        dense_features = None

    dev_crf_examples = [
        prepare_example(
            ex,
            crf_extractor=crf_extractor,
            tokenizer=tokenizer,
            dense_features=dense_features,
        )
        for ex in dev_examples
    ]

    f1_score, classification_report = crf_extractor.eval(dev_crf_examples)
    msg.warn(f"f1 score: {f1_score}")
    print(classification_report)


if __name__ == "__main__":
    plac.call(main)
