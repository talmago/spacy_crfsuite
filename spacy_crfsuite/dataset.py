import json
import logging
import os
import re

from typing import List, Dict, Optional, Text

from spacy_crfsuite.bilou import get_entity_offsets
from spacy_crfsuite.crf_extractor import CRFToken, CRFExtractor
from spacy_crfsuite.tokenizer import SpacyTokenizer

LOG = logging.getLogger("dataset")

# regex for: - XXX
_item_regex = re.compile(r"\s*[-*+]\s*(.+)")

# regex for: `[entity_text](entity_type(:entity_synonym)?)`
ent_regex = re.compile(
    r"\[(?P<entity_text>[^\]]+)" r"\]\((?P<entity>[^:)]*?)" r"(?:\:(?P<value>[^)]+))?\)"
)

# regex for comments in markdown
_comment_regex = re.compile(r"<!--[\s\S]*?--!*>", re.MULTILINE)


def read_file(fname) -> List[Dict]:
    _, ext = os.path.splitext(fname)
    if ext.lower() == ".json":
        with open(fname, "r") as f:
            return json.load(f)
    elif ext.lower() == ".md":
        with open(fname, "r") as f:
            content = f.read()
            return read_markdown(content)
    else:
        raise ValueError(f"expected a JSON / Markdown, got {ext}.")


def create_dataset(
    examples: List[Dict], tokenizer: Optional[SpacyTokenizer] = None
) -> List[List[CRFToken]]:
    dataset = []
    crf_extractor = CRFExtractor()
    tokenizer = tokenizer or SpacyTokenizer()

    for example in examples:
        if "tokens" in example:
            pass
        elif "text" in example:
            example["tokens"] = tokenizer.tokenize(example, attribute="text")
        else:
            LOG.warning("Empty message: %s", example)
            continue

        entity_offsets = get_entity_offsets(example)
        entities = crf_extractor.from_json_to_crf(example, entity_offsets)
        dataset.append(entities)

    return dataset


def read_markdown(s: Text, sections: Optional[List[Text]] = None) -> List[Dict]:
    """Read markdown string and create TrainingData object"""
    current_section = None
    training_examples = []
    s = _strip_comments(s)
    for line in s.splitlines():
        line = line.strip()
        header = _find_section_header(line)
        if header:
            current_section = header
        elif sections is None or (sections and current_section in sections):
            message = _parse_item(line)
            training_examples.append(message)
    return training_examples


def _strip_comments(text: Text) -> Text:
    """ Removes comments defined by `comment_regex` from `text`. """
    return re.sub(_comment_regex, "", text)


def _find_section_header(line: Text) -> Optional[Text]:
    """Checks if the current line contains a section header
    and returns the section and the title."""
    match = re.search(r"##\s*(.+)?", line)
    if match is not None:
        return match.group(1)
    return


def _parse_item(line: Text) -> Optional[Dict]:
    """Parses an md list item line based on the current section type."""
    match = re.match(_item_regex, line)
    if match:
        example = match.group(1)
        entities = _find_entities_in_training_example(example)
        plain_text = re.sub(ent_regex, lambda m: m.groupdict()["entity_text"], example)
        return {"text": plain_text, "entities": entities}
    return


def _find_entities_in_training_example(example: Text) -> List[Dict]:
    """Extracts entities from a markdown intent example."""
    entities = []
    offset = 0
    for match in re.finditer(ent_regex, example):
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
