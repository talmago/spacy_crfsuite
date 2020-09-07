from typing import List, Tuple, Text, Optional, Dict, Any

from spacy_crfsuite.tokenizer import Token

BILOU_PREFIXES = {"B-", "I-", "U-", "L-"}
NO_ENTITY_TAG = "O"


def get_entity_offsets(message: Dict) -> List[Tuple[int, int, Text]]:
    """Maps the entities of the given message to their start, end, and tag values.

    Args:
        message: the message

    Returns:
        a list of start, end, and tag value tuples
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

    Returns:
        the tag without the BILOU prefix
    """
    if tag[:2] in BILOU_PREFIXES:
        return tag[2:]
    return tag


def bilou_tags_from_offsets(
    tokens: List[Token], entities: List[Tuple[int, int, Text]]
) -> List[Text]:
    """Creates a list of BILOU tags for the given list of tokens and entities.

    Args:
        tokens: the list of tokens
        entities: the list of start, end, and tag tuples.

    Returns:
        BILOU tags.
    """
    start_pos_to_token_idx = {token.start: i for i, token in enumerate(tokens)}
    end_pos_to_token_idx = {token.end: i for i, token in enumerate(tokens)}
    bilou = [NO_ENTITY_TAG for _ in tokens]

    _add_bilou_tags_to_entities(
        bilou, entities, end_pos_to_token_idx, start_pos_to_token_idx
    )

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


def remove_bilou_prefixes(bilou: List[Text]):
    for i, label in enumerate(bilou):
        if bilou_prefix_from_tag(label):  # removes BILOU prefix from label
            bilou[i] = entity_name_from_tag(label)
