import numpy as np

from typing import Text, Dict, Any, Optional, List, Union, NamedTuple
from wasabi import msg

from spacy_crfsuite.bilou import get_entity_offsets, bilou_tags_from_offsets
from spacy_crfsuite.tokenizer import Token


class Pooling:
    MEAN = "mean"
    MAX = "max"


class CRFToken(NamedTuple):
    text: Text
    tag: Text
    entity: Text
    shape: Union[Text, int]
    pattern: Dict[Text, Any]
    dense_features: np.ndarray


class Featurizer:
    """Translate text into a list of CRF tokens using the BILOU schema.

    It must be called after the pre-processing step,
    either by spaCy or external source (ConLL file)."""

    cfg: Dict[str, Any] = {
        "use_dense_features": False,
        "dense_features_cls_pooling": Pooling.MEAN,
    }

    def __init__(self, **overrides):
        self.cfg.update(overrides)

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
        text_dense_features = self.get_dense_features(message)
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

    def get_dense_features(self, message: Dict) -> Optional[List[Any]]:
        """Compute dense features for input message.

        Args:
            message (dict): message dict.

        Returns:
            list of dense features.
        """
        if not self.cfg["use_dense_features"]:
            return None

        tokens = self.tokens_without_cls(message)
        features = [t.get("vector") for t in tokens if t.get("vector") is not None]
        if len(features) > 0:
            features = np.vstack(features)
            pooling = self.cfg["dense_features_cls_pooling"]
            cls_token_vec = self._calculate_cls_vector(features, pooling)
            features = np.concatenate([features, cls_token_vec])
        if len(features) != len(tokens) + 1:
            return None
        # convert to python-crfsuite feature format
        features_out = []
        for feature in features:
            feature_dict = {
                str(index): token_features for index, token_features in enumerate(feature)
            }
            converted = {"text_dense_features": feature_dict}
            features_out.append(converted)
        return features_out

    def apply_bilou_schema(self, message: Dict) -> List[Text]:
        """Apply BILOU schema to a gold standard JSON example.

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
        return message.get("tokens")[:-1]

    @staticmethod
    def __pattern_of_token(message: Dict, i: int) -> Dict:
        if message.get("tokens") is not None:
            return message.get("tokens")[i].get("pattern", {})
        else:
            return {}

    @staticmethod
    def _calculate_cls_vector(
        features: np.ndarray, pooling: Text = Pooling.MEAN
    ) -> np.ndarray:
        # take only non zeros feature vectors into account
        non_zero_features = np.array([f for f in features if f.any()])
        # if features are all zero just return a vector with all zeros
        if non_zero_features.size == 0:
            return np.zeros([1, features.shape[-1]])
        if pooling == Pooling.MEAN:
            return np.mean(non_zero_features, axis=0, keepdims=True)
        elif pooling == Pooling.MAX:
            return np.max(non_zero_features, axis=0, keepdims=True)
        else:
            raise ValueError(
                f"Invalid pooling operation specified. Available operations are "
                f"'{Pooling.MEAN}' or '{Pooling.MAX}', but provided value is "
                f"'{pooling}'."
            )
