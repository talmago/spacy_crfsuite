import plac
import srsly

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from wasabi import msg

from spacy_crfsuite import CRFExtractor, to_crfsuite
from spacy_crfsuite.utils import read_examples


@plac.annotations(
    in_file=("Path to input file (either .json, .md or .conll)", "positional", None, str),
    model_file=("Path to model file", "option", "m", str),
    config_file=("Path to config file (.json format)", "option", "c", str),
)
def main(in_file, model_file=None, config_file=None):
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
    dev_examples = read_examples(in_file)
    num_dev_examples = len(dev_examples)
    dev = to_crfsuite(dev_examples, crf_extractor=crf_extractor)

    msg.good(f"Successfully loaded {num_dev_examples} dev examples.")
    f1_score, classification_report = crf_extractor.eval(dev)
    msg.warn(f"f1 score: {f1_score}")
    print(classification_report)


if __name__ == "__main__":
    plac.call(main)
