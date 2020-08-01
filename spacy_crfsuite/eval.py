from typing import List, Dict, Optional

import plac
import srsly
import warnings

from wasabi import msg

from spacy_crfsuite.bilou import get_entity_offsets
from spacy_crfsuite.tokenizer import SpacyTokenizer, Tokenizer

warnings.simplefilter(action="ignore", category=FutureWarning)

from spacy_crfsuite import CRFExtractor, CRFToken
from spacy_crfsuite.utils import read_examples


def to_crfsuite(
    examples: List[Dict],
    crf_extractor: Optional[CRFExtractor] = None,
    tokenizer: Optional[SpacyTokenizer] = None,
) -> List[List[CRFToken]]:
    """Translate training examples to CRF features.

    Args:
        examples (list): training examples.
        crf_extractor (CRFExtractor): crf component.
        tokenizer (Tokenizer): optional, tokenizer. Default is `SpacyTokenizer`.

    Returns:
        List[List[CRFToken]], CRF dataset.
    """
    tokenizer = tokenizer or SpacyTokenizer()
    assert isinstance(tokenizer, Tokenizer)

    crf_extractor = crf_extractor or CRFExtractor()
    assert isinstance(crf_extractor, CRFExtractor)

    dataset = []
    for example in examples:
        if not example:
            continue
        if "tokens" in example:
            pass
        elif "text" in example:
            example["tokens"] = tokenizer.tokenize(example, attribute="text")
        else:
            try:
                from wasabi import msg

                msg.warn(f"Empty example: {example}")
            except ImportError:
                pass

            continue
        entity_offsets = get_entity_offsets(example)
        entities = crf_extractor.from_json_to_crf(example, entity_offsets)
        dataset.append(entities)

    return dataset


@plac.annotations(
    in_file=(
        "Path to input file (either .json, .md or .conll)",
        "positional",
        None,
        str,
    ),
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
