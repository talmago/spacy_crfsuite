import copy
import re

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


def read_examples(path: Union[Path, str]) -> List[Dict]:
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
        return list(srsly.read_json(path))

    elif ext == ".jsonl":
        return list(srsly.read_jsonl(path))

    elif ext in (".md", ".markdown"):
        with path.open("r", encoding="utf-8") as f:
            md_reader = MarkdownReader()
            return md_reader(f.read())

    else:
        raise ValueError(f"expected a JSON / Markdown, got {ext}.")


class MarkdownReader:
    """A class to read MD format and translate (in-memory) into JSON format.
    """

    item_regex = re.compile(r"\s*[-*+]\s*(.+)")
    ent_regex = re.compile(
        r"\[(?P<entity_text>[^\]]+)"
        r"\]\((?P<entity>[^:)]*?)"
        r"(?:\:(?P<value>[^)]+))?\)"
    )
    comment_regex = re.compile(r"<!--[\s\S]*?--!*>", re.MULTILINE)

    def __call__(self, text: Text, headers: Optional[List[Text]] = None) -> List[Dict]:
        """Read markdown string and create TrainingData object"""
        training_examples = []
        current_section = None
        text = self.strip_comments(text)
        for line in text.splitlines():
            line = line.strip()
            header = self.find_section_header(line)
            if header:
                current_section = header
            elif headers is None or (headers and current_section in headers):
                message = self.parse_item(line)
                if message:
                    training_examples.append(message)
        return training_examples

    def parse_item(self, line: Text) -> Optional[Dict]:
        """Parses an md list item line based on the current section type."""
        match = re.match(MarkdownReader.item_regex, line)
        if match:
            example = match.group(1)
            entities = self.find_entities_in_training_example(example)
            plain_text = re.sub(
                MarkdownReader.ent_regex, lambda m: m.groupdict()["entity_text"], example
            )
            return {"text": plain_text, "entities": entities}

    @staticmethod
    def strip_comments(text: Text) -> Text:
        """Removes comments defined by `comment_regex` from `text`."""
        return re.sub(MarkdownReader.comment_regex, "", text)

    @staticmethod
    def find_section_header(line: Text) -> Optional[Text]:
        """Checks if the current line contains a section header
        and returns the section and the title."""
        match = re.search(r"##\s*(.+)?", line)
        if match is not None:
            return match.group(1)

    @staticmethod
    def find_entities_in_training_example(example: Text) -> List[Dict]:
        """Extracts entities from a markdown intent example."""
        entities = []
        offset = 0
        for match in re.finditer(MarkdownReader.ent_regex, example):
            entity_text = match.groupdict()["entity_text"]
            entity_type = match.groupdict()["entity"]
            if match.groupdict()["value"]:
                entity_value = match.groupdict()["value"]
            else:
                entity_value = entity_text
            start_index = match.start() - offset
            end_index = start_index + len(entity_text)
            offset += len(match.group(0)) - len(entity_text)
            entity = {
                "start": start_index,
                "end": end_index,
                "value": entity_value,
                "entity": entity_type,
            }
            entities.append(entity)
        return entities
