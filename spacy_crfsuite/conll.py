from pathlib import Path
from typing import List, TextIO, Iterator, Dict, Union
from spacy_crfsuite.tokenizer import Token


def read_conll(path: Union[str, Path]) -> Iterator[Dict]:
    """Read a CONLL file.

    Args:
        path: file path.

    Returns:
        Iterator.
    """
    if not isinstance(path, Path):
        path = Path(path)

    assert isinstance(path, Path)

    with path.open("r", encoding="utf-8") as f:
        yield from _parse_conll(f)


def _parse_conll(in_file: TextIO) -> Iterator[Dict]:
    """Parse a text blob of CONLL format.

    https://github.com/EmilStenstrom/conllu/blob/bd22e8680ec12b9f676e755c82f32517f5a399e1/conllu/parser.py#L53"""
    buf: List[List[str]] = []
    for line in in_file:
        if line == "\n":
            if not buf:
                continue
            yield _parse_block(buf)
            buf = []
        elif line.startswith("-DOCSTART-"):
            continue
        elif line.startswith("#"):
            continue
        else:
            buf.append(line.rstrip().split())
    if buf:
        yield _parse_block(buf)


def _parse_block(buff: List[List[str]]) -> Dict:
    """Parse a buffer in CONLL file.

    Args:
        buff: splitted lines.

    Returns:
        dict
    """
    tokens = []
    entities = []
    idx = 0

    for i, pieces in enumerate(buff):
        if len(pieces) == 3:  # conll_02
            word, pos, tag = pieces
        elif len(pieces) == 4:  # conll_03
            word, pos, pos2, tag = pieces
        else:
            raise ValueError(f"Could not parse: {buff[i]}")
        token = Token(word, idx, data={"pos": pos})
        tokens.append(token)
        idx += len(word) + 1
        if tag != "O":
            entities.append(
                {
                    "value": token.text,
                    "entity": tag,
                    "start": token.start,
                    "end": token.end,
                }
            )

    return {
        "text": " ".join(token.text for token in tokens),
        "tokens": tokens,
        "entities": entities,
    }
