import pathlib

from typing import List, Dict, Optional

import plac
import srsly
from wasabi import msg

from spacy_crfsuite.bilou import get_entity_offsets
from spacy_crfsuite.crf_extractor import CRFToken, CRFExtractor
from spacy_crfsuite.tokenizer import SpacyTokenizer, Tokenizer
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
    out_dir=("Path to output directory", "option", "o", str),
    config_file=("Path to config file (.json format)", "option", "c", str),
)
def main(in_file, out_dir=None, config_file=None):
    """Train CRF entity tagger."""
    if config_file:
        msg.info(f"Loading config: {config_file}")
        component_config = srsly.read_json(config_file)
    else:
        component_config = None

    train_examples = read_examples(in_file)
    num_train_examples = len(train_examples)

    msg.info(f"Training CRF entity tagger with {num_train_examples} examples.")
    crf_extractor = CRFExtractor(component_config=component_config)

    train_dataset = to_crfsuite(train_examples, crf_extractor=crf_extractor)
    crf_extractor.train(train_dataset)

    model_path = pathlib.Path(out_dir or ".").resolve() / "model.pkl"
    msg.info("Saving model to disk")
    model_path.parent.mkdir(exist_ok=True)
    crf_extractor.to_disk(model_path)
    msg.good("Successfully saved model to file.", str(model_path.resolve()))


if __name__ == "__main__":
    plac.call(main)
