import os
import pathlib
import plac
import spacy
import srsly

from typing import Optional, Dict, List
from wasabi import msg

from spacy_crfsuite.bilou import remove_bilou_prefixes
from spacy_crfsuite.crf_extractor import CRFExtractor
from spacy_crfsuite.features import Featurizer, CRFToken
from spacy_crfsuite.tokenizer import SpacyTokenizer, Tokenizer
from spacy_crfsuite.utils import read_file


def gold_example_to_crf_tokens(
    example: Dict,
    tokenizer: Optional[Tokenizer] = None,
    use_dense_features: bool = False,
    bilou: bool = True,
) -> List[CRFToken]:
    """Translate training example to CRF feature space.

    Args:
        example (dict): example dict. must have either "doc", "tokens" or "text" field.
        tokenizer (Tokenizer): tokenizer.
        use_dense_features (bool): use dense features.
        bilou (bool): apply BILOU tags to example.

    Returns:
        List[CRFToken], CRF example.
    """
    if not example:
        return []

    tokenizer = tokenizer or SpacyTokenizer()
    featurizer = Featurizer(use_dense_features=use_dense_features)

    if "tokens" in example:
        # tokenized by 3rd party, nothing to do .. except for dense feature addition (when needed)
        if use_dense_features and isinstance(tokenizer, SpacyTokenizer):
            for token in example["tokens"]:
                vector = tokenizer.get_vector(token)
                if vector is not None:
                    token.set("vector", vector)

    elif "text" in example:
        # Call a tokenizer to tokenize the message. Default is SpacyTokenizer.
        tokenizer.tokenize(example, attribute="text")
    else:
        raise ValueError(
            f"Bad example: {example}. " f"Attribute ``text`` or ``tokens`` is missing."
        )
    # By default, JSON examples don't have a tagging schema like "BILOU".
    # If they do, like in CoNLL datasets, we strip them after alignment.
    entities = featurizer.apply_bilou_schema(example)
    if not bilou:
        remove_bilou_prefixes(entities)
    return featurizer(example, entities)


@plac.annotations(
    in_file=("Path to input file (either .json, .md or .conll)", "positional", None, str),
    model_file=("Path to model file", "option", "m", str),
    out_dir=("Path to output directory", "option", "o", str),
    config_file=("Path to config file (.json format)", "option", "c", str),
    spacy_model=("Name of spaCy model to use", "option", "lm", str),
    fine_tune=("Fine tune hyper parameters before training.", "flag", "ft", bool),
)
def main(
    in_file,
    out_dir=None,
    model_file=None,
    config_file=None,
    spacy_model=None,
    fine_tune=False,
):
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
    use_dense_features = crf_extractor.use_dense_features()
    train_crf = [
        gold_example_to_crf_tokens(
            ex, tokenizer=tokenizer, use_dense_features=use_dense_features
        )
        for ex in train_examples
    ]

    if fine_tune:
        msg.info("Fine-tuning hyper params.")
        rs = crf_extractor.fine_tune(train_crf, cv=5, n_iter=30, random_state=42)
        msg.good("Setting fine-tuned hyper params:", rs.best_params_)
        crf_extractor.component_config.update(rs.best_params_)

    msg.info("Training entity tagger with CRF.")
    crf_extractor.train(train_crf)

    model_path = pathlib.Path(out_dir or ".").resolve() / "model.pkl"
    msg.info("Saving model to disk")
    model_path.parent.mkdir(exist_ok=True)
    crf_extractor.to_disk(model_path)
    msg.good("Successfully saved model to file.", str(model_path.relative_to(os.getcwd())))


if __name__ == "__main__":
    plac.call(main)
