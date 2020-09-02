import os
import pathlib
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
    out_dir=("Path to output directory", "option", "o", str),
    config_file=("Path to config file (.json format)", "option", "c", str),
    spacy_model=("Name of spaCy model to use", "positional", None, str),
)
def main(in_file, out_dir=None, model_file=None, config_file=None, spacy_model=None):
    """Train CRF entity tagger."""
    if config_file:
        msg.info("Loading config from disk")
        component_config = srsly.read_json(config_file)
        msg.good("Successfully loaded config from file.", config_file)
    else:
        component_config = None

    crf_extractor = CRFExtractor(component_config=component_config)

    if model_file is not None:
        msg.info(f"Loading model from disk.")
        crf_extractor = crf_extractor.from_disk(model_file)
        msg.good("Successfully loaded model from file.", model_file)

    msg.info("Loading training examples.")
    train_examples = read_file(in_file)
    msg.good(
        f"Successfully loaded {len(train_examples)} training examples from file.", in_file
    )

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

    train_crf_examples = [
        prepare_example(
            ex,
            crf_extractor=crf_extractor,
            tokenizer=tokenizer,
            dense_features=dense_features,
        )
        for ex in train_examples
    ]

    msg.info("Training entity tagger with CRF.")
    crf_extractor.train(train_crf_examples)

    model_path = pathlib.Path(out_dir or ".").resolve() / "model.pkl"
    msg.info("Saving model to disk")
    model_path.parent.mkdir(exist_ok=True)
    crf_extractor.to_disk(model_path)
    msg.good("Successfully saved model to file.", str(model_path.relative_to(os.getcwd())))


if __name__ == "__main__":
    plac.call(main)
