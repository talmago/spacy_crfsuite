import re
from typing import Text, Optional, List, Dict


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
