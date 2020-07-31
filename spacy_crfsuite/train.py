import pathlib
import plac
import srsly

from wasabi import msg

from spacy_crfsuite import CRFExtractor
from spacy_crfsuite.dataset import read_file, create_dataset


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

    train_examples = read_file(in_file)
    num_train_examples = len(train_examples)
    train_dataset = create_dataset(train_examples)

    msg.info(f"Training CRF entity tagger with {num_train_examples} examples.")
    crf_extractor = CRFExtractor(component_config=component_config)
    crf_extractor.train(train_dataset)

    model_path = pathlib.Path(out_dir or ".") / "model.pkl"
    msg.info("Saving model to disk")
    model_path.mkdir(exist_ok=True)
    crf_extractor.to_disk(model_path)
    msg.good("Successfully saved model to file.", str(model_path.resolve()))


if __name__ == "__main__":
    plac.call(main)
