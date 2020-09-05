import os
import pathlib
import plac
import spacy
import srsly

from wasabi import msg
from typing import Optional, Dict, List

from spacy_crfsuite.crf_extractor import CRFExtractor
from spacy_crfsuite.dense_features import DenseFeatures
from spacy_crfsuite.features import Featurizer, CRFToken
from spacy_crfsuite.tokenizer import SpacyTokenizer, Tokenizer
from spacy_crfsuite.utils import read_file


def crf_tokens(
    example: Dict,
    featurizer: Optional[Featurizer] = None,
    tokenizer: Optional[Tokenizer] = None,
    dense_features: Optional[DenseFeatures] = None,
    apply_bilou: bool = True,
) -> List[CRFToken]:
    """Translate training example to CRF feature space.

    Args:
        example (dict): example dict. must have either "doc", "tokens" or "text" field.
        featurizer (Featurizer): featurizer.
        tokenizer (Tokenizer): tokenizer.
        dense_features (DenseFeatures): dense features.
        apply_bilou (bool): apply the bilou schema (used for gold standard example).

    Returns:
        List[CRFToken], CRF example.
    """
    if not example:
        return []

    if "tokens" in example:
        # tokenized by 3rd party, nothing to do ..
        pass
    elif "text" in example:
        # Call a tokenizer to tokenize the message. Default is SpacyTokenizer.
        tokenizer = tokenizer or SpacyTokenizer()
        example["tokens"] = tokenizer.tokenize(example, attribute="text")
    else:
        raise ValueError(
            f"Bad example: {example}. " f"Attribute ``text`` or ``tokens`` is missing."
        )

    if dense_features:
        text_dense_features = dense_features(
            example, attribute="doc" if "doc" in example else "tokens"
        )
        if len(text_dense_features) > 0:
            example["text_dense_features"] = text_dense_features

    featurizer = featurizer or Featurizer()
    entities = featurizer.apply_bilou_schema(example) if apply_bilou else None
    return featurizer(example, entities)


@plac.annotations(
    in_file=("Path to input file (either .json, .md or .conll)", "positional", None, str),
    model_file=("Path to model file", "option", "m", str),
    out_dir=("Path to output directory", "option", "o", str),
    config_file=("Path to config file (.json format)", "option", "c", str),
    spacy_model=("Name of spaCy model to use", "option", "lm", str),
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
        crf_tokens(ex, tokenizer=tokenizer, dense_features=dense_features)
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
