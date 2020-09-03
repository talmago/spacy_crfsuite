from typing import NamedTuple, Text, Dict, Any, Optional, List

import numpy as np
from wasabi import msg

from spacy_crfsuite.bilou import get_entity_offsets, bilou_tags_from_offsets
from spacy_crfsuite.constants import TOKENS, PATTERN, DENSE_FEATURES
from spacy_crfsuite.tokenizer import Token


class CRFToken(NamedTuple):
    text: Text
    tag: Text
    entity: Text
    shape: Text
    pattern: Dict[Text, Any]
    dense_features: np.ndarray


class Featurizer:
    """Translate text into a list of CRF tokens using the BILOU schema.

     It must be called after the pre-processing step,
     either by spaCy or external source (ConLL file)."""

    def __call__(
        self, message: Dict, entities: Optional[List[Text]] = None
    ) -> List[CRFToken]:
        """Convert JSON example to crfsuite format.

        Args:
            message (dict): message dict.
            entities (list): optional, GOLD labels for entities.

        Returns:
            a list of CRF tokens.
        """
        crf_tokens = []
        tokens = self.tokens_without_cls(message)
        text_dense_features = self.__get_dense_features(message)
        for i, token in enumerate(tokens):
            pattern = self.__pattern_of_token(message, i)
            entity = entities[i] if entities else "N/A"
            pos = token.get("pos")
            shape = token.get("shape")
            dense_features = (
                text_dense_features[i] if text_dense_features is not None else []
            )
            crf_token = CRFToken(token.text, pos, entity, shape, pattern, dense_features)
            crf_tokens.append(crf_token)
        return crf_tokens

    def apply_bilou_schema(self, message: Dict) -> List[Text]:
        """Apply BILOU schema to a JSON example.

        Args:
            message (dict): message dict.

        Returns:
            a list of BILOU tags.
        """
        tokens = self.tokens_without_cls(message)
        entity_offsets = get_entity_offsets(message)
        entities = bilou_tags_from_offsets(tokens, entity_offsets)

        collected = []
        for t, e in zip(tokens, entities):
            if e == "-":
                collected.append(t)
            elif collected:
                collected_text = " ".join([t.text for t in collected])
                msg and msg.warn(
                    f"Misaligned entity annotation for '{collected_text}' "
                    f"in sentence: \"{message['text']}\". "
                    f"Make sure the start and end values of the "
                    f"annotated training examples end at token "
                    f"boundaries (e.g. don't include trailing "
                    f"whitespaces or punctuation)."
                )
                collected = []

        return entities

    @staticmethod
    def tokens_without_cls(message: Dict) -> List[Token]:
        return message.get(TOKENS)[:-1]

    @staticmethod
    def __pattern_of_token(message: Dict, i: int) -> Dict:
        if message.get(TOKENS) is not None:
            return message.get(TOKENS)[i].get(PATTERN, {})
        else:
            return {}

    @staticmethod
    def __get_dense_features(message: Dict) -> Optional[List[Any]]:
        features = message.get(DENSE_FEATURES)
        if features is None:
            return None

        tokens = message.get(TOKENS, [])
        if len(tokens) != len(features):
            msg and msg.warning(
                f"Number of features ({len(features)}) for attribute "
                f"'{DENSE_FEATURES}' "
                f"does not match number of tokens ({len(tokens)}). Set "
                f"'return_sequence' to true in the corresponding featurizer in order "
                f"to make use of the features in 'CRFEntityExtractor'."
            )
            return None

        # convert to python-crfsuite feature format
        features_out = []
        for feature in features:
            feature_dict = {
                str(index): token_features for index, token_features in enumerate(feature)
            }
            converted = {DENSE_FEATURES: feature_dict}
            features_out.append(converted)
        return features_out