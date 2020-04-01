import argparse
import json
import logging
import os

from typing import Text, Optional

from spacy_crfsuite import CRFExtractor
from spacy_crfsuite.dataset import read_file, create_dataset

LOG = logging.getLogger("trainer")


def _load_config(config_file: Optional[Text]):
    if config_file:
        with open(config_file, "r") as f:
            component_config = json.load(f)
    else:
        component_config = None

    return component_config


def train(
    train_file: Text,
    model_dir: Text,
    model_name: Text = "crf",
    config_file: Optional[Text] = None,
) -> None:
    LOG.info("Loading dataset")
    train_examples = read_file(train_file)
    train = create_dataset(train_examples)

    crf_extractor = CRFExtractor(component_config=_load_config(config_file))
    LOG.info("Train CRF model with %d examples.", len(train_examples))
    crf_extractor.train(train)

    LOG.info("Save model %r to %r", model_name, model_dir)
    os.makedirs(model_dir, exist_ok=True)
    crf_extractor.persist(model_dir, model_name)


def eval(
    eval_file: Text,
    model_dir: Text,
    model_name: Text = "crf",
    config_file: Optional[Text] = None,
) -> None:
    dev_examples = read_file(eval_file)
    LOG.info("Dev examples: %d", len(dev_examples))
    dev = create_dataset(dev_examples)

    LOG.info("Load model %r from %r", model_name, model_dir)
    crf_extractor = CRFExtractor.load(
        model_dir=model_dir,
        model_name=model_name,
        component_config=_load_config(config_file),
    )

    f1_score, classification_report = crf_extractor.eval(dev)
    LOG.info("f1_score: %1.2f", f1_score)
    LOG.info("classification report\n%s", classification_report)


def parse_arguments():
    parser = argparse.ArgumentParser(description="train CRF extractor")

    parser.add_argument(
        "command", choices=["train", "eval"], default=None, help="command to run"
    )

    parser.add_argument("input_file", default=None, help="data set (JSON / markdown)")

    parser.add_argument(
        "--model_dir",
        default=os.getcwd(),
        help="path to a directory where model will be saved.",
    )

    parser.add_argument(
        "--model_name",
        default="crf",
        help="model will be saved as ``MODEL_NAME.pkl``. default is 'crf'.",
    )

    parser.add_argument(
        "--config",
        default=None,
        help="optional configuration file for CRFExtractor class.",
    )

    options = parser.parse_args()

    if not os.path.isfile(options.input_file):
        raise IOError(f"-E- file not found: {options.input_file}")

    return options


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)-15s -- %(levelname)s -- %(module)s] %(message)s",
        level=logging.INFO,
    )

    args = parse_arguments()

    if args.command == "train":
        train(
            args.input_file,
            model_dir=args.model_dir,
            model_name=args.model_name,
            config_file=args.config,
        )

    eval(
        args.input_file,
        model_dir=args.model_dir,
        model_name=args.model_name,
        config_file=args.config,
    )
