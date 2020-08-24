from pathlib import Path
from typing import Union, Generator

from spacy_crfsuite.bilou import bilou_prefix_from_tag
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
        sent_strs = file_.read().strip().replace("\n  \n", "\n\n").split("\n\n")

    for sent_str in sent_strs:
        tokens = []
        entities = []
        idx = 0
        lines = [li.split() for li in sent_str.split("\n") if not li.startswith("#")]
        for i, pieces in enumerate(lines):
            lemma = None
            if len(pieces) == 3:  # conll_02
                word, pos, tag = pieces
            elif len(pieces) == 4:  # conll_03
                word, pos, pos2, tag = pieces
            else:
                continue
            token = Token(word, idx, lemma=lemma, data={"pos": pos})
            tokens.append(token)
            idx += len(word) + 1
            if bilou_prefix_from_tag(tag):
                entities.append(
                    {
                        "value": token.text,
                        "entity": tag,
                        "start": token.start,
                        "end": token.end,
                    }
                )

        yield {
            "text": " ".join(token.text for token in tokens),
            "tokens": tokens,
            "entities": entities,
        }
