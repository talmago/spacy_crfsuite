from pathlib import Path
from typing import Union, Generator

from spacy_crfsuite.tokenizer import Token


def read_conll(loc: Union[str, Path]) -> Generator:
    """Read a CONLL file.

    Args:
        loc: file location.

    Returns:
        Generator.
    """
    if not isinstance(loc, Path):
        loc = Path(loc)
    assert isinstance(loc, Path)

    with Path(loc).open(encoding="utf8") as file_:
        sent_strs = file_.read().strip().split("\n\n")

    for sent_str in sent_strs:
        lines = [li.split() for li in sent_str.split("\n") if not li.startswith("#")]
        tokens = []
        labels = []
        for i, pieces in enumerate(lines):
            lemma = None
            if len(pieces) == 4:
                word, pos, head, label = pieces
            else:
                idx, word, lemma, pos1, pos, morph, head, label, _, _2 = pieces
            if "-" in idx:
                continue
            token = Token(word, idx, lemma=lemma, data={"pos": pos})
            tokens.append(token)
            tags.append(label)
        yield tokens, labels


if __name__ == "__main__":
    for words, tags in read_conll(
        "/Users/talmago/git/entitytargeting/data/conll/train.conll"
    ):
        print(words)
        break
