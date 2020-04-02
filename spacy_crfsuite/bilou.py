from typing import List, Tuple, Text, Optional, Dict, Set, Any

from spacy_crfsuite.tokenizer import Token

BILOU_PREFIXES = ["B-", "I-", "U-", "L-"]


def get_entity_offsets(message: Dict) -> List[Tuple[int, int, Text]]:
    """Maps the entities of the given message to their start, end, and tag values.

    Args:
        message: the message

    Returns: a list of start, end, and tag value tuples
    """

    def convert_entity(entity: Dict[Text, Any]) -> Tuple[int, int, Text]:
        return entity["start"], entity["end"], entity["entity"]

    return [convert_entity(entity) for entity in message.get("entities", [])]


def bilou_prefix_from_tag(tag: Text) -> Optional[Text]:
    """Returns the BILOU prefix from the given tag.
    Args:
        tag: the tag
    Returns: the BILOU prefix of the tag
    """
    if tag[:2] in BILOU_PREFIXES:
        return tag[0]
    return None


def entity_name_from_tag(tag: Text) -> Text:
    """Remove the BILOU prefix from the given tag.
    Args:
        tag: the tag
    Returns: the tag without the BILOU prefix
    """
    if tag[:2] in BILOU_PREFIXES:
        return tag[2:]
    return tag


def bilou_tags_from_offsets(
    tokens: List[Token], entities: List[Tuple[int, int, Text]], missing: Text = "O"
) -> List[Text]:
    """Creates a list of BILOU tags for the given list of tokens and entities.
    Args:
        tokens: the list of tokens
        entities: the list of start, end, and tag tuples
        missing: tag for missing entities
    Returns: a list of BILOU tags
    """
    start_pos_to_token_idx = {token.start: i for i, token in enumerate(tokens)}
    end_pos_to_token_idx = {token.end: i for i, token in enumerate(tokens)}
    bilou = ["-" for _ in tokens]

    # Handle entity cases
    _add_bilou_tags_to_entities(
        bilou, entities, end_pos_to_token_idx, start_pos_to_token_idx
    )

    # Now distinguish the O cases from ones where we miss the tokenization
    entity_positions = _get_entity_positions(entities)
    _handle_not_an_entity(bilou, tokens, entity_positions, missing)

    return bilou


def _add_bilou_tags_to_entities(
    bilou: List[Text],
    entities: List[Tuple[int, int, Text]],
    end_pos_to_token_idx: Dict[int, int],
    start_pos_to_token_idx: Dict[int, int],
):
    for start_pos, end_pos, label in entities:
        start_token_idx = start_pos_to_token_idx.get(start_pos)
        end_token_idx = end_pos_to_token_idx.get(end_pos)

        # Only interested if the tokenization is correct
        if start_token_idx is not None and end_token_idx is not None:
            if start_token_idx == end_token_idx:
                bilou[start_token_idx] = f"U-{label}"
            else:
                bilou[start_token_idx] = f"B-{label}"
                for i in range(start_token_idx + 1, end_token_idx):
                    bilou[i] = f"I-{label}"
                bilou[end_token_idx] = f"L-{label}"


def _get_entity_positions(entities: List[Tuple[int, int, Text]]) -> Set[int]:
    entity_positions = set()

    for start_pos, end_pos, label in entities:
        for i in range(start_pos, end_pos):
            entity_positions.add(i)

    return entity_positions


def _handle_not_an_entity(
    bilou: List[Text], tokens: List[Token], entity_positions: Set[int], missing: Text
):
    for n, token in enumerate(tokens):
        for i in range(token.start, token.end):
            if i in entity_positions:
                break
        else:
            bilou[n] = missing
