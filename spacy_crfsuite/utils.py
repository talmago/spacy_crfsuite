import copy
from pathlib import Path
from typing import Optional, Dict, Text, Any, List, Union
import srsly


def override_defaults(
    defaults: Optional[Dict[Text, Any]], custom: Optional[Dict[Text, Any]]
) -> Dict[Text, Any]:
    if defaults:
        cfg = copy.deepcopy(defaults)
    else:
        cfg = {}

    if custom:
        for key in custom.keys():
            if isinstance(cfg.get(key), dict):
                cfg[key].update(custom[key])
            else:
                cfg[key] = custom[key]

    return cfg


def read_file(path: Union[Path, str]) -> List[Dict]:
    """Read train/dev examples from file, either JSON, MD or ConLL format.

    Args:
        path: file path.

    Returns:
        list of examples
    """
    if not isinstance(path, Path):
        path = Path(path)
    assert isinstance(path, Path)

    ext = path.suffix.lower()

    if ext == ".json":
        # JSON format is the GOLD standard ...
        return list(srsly.read_json(path))

    elif ext == ".jsonl":
        # same here ..
        return list(srsly.read_jsonl(path))

    elif ext in (".md", ".markdown"):
        from spacy_crfsuite.markdown import MarkdownReader

        # With markdown, we can easily convert to JSON
        with path.open("r", encoding="utf-8") as f:
            md_reader = MarkdownReader()
            return md_reader(f.read())

    elif ext in (".txt", ".conll"):
        from spacy_crfsuite.conll import read_conll

        # CoNLL-02, CoNLL-03
        return list(read_conll(path))

    else:
        raise ValueError(
            f"Can't read examples from file with extension: ({ext}). "
            f"spacy_crfsuite accepts .json, .jsonl, .txt, .conll files."
        )
