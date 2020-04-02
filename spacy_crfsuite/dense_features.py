import numpy as np
import spacy

from typing import Text, Dict, Any

from spacy_crfsuite.constants import DENSE_FEATURES


class Pooling:
    MEAN = "mean"
    MAX = "max"


class DenseFeatures:
    def __init__(self, nlp, pooling: Text = Pooling.MEAN):
        assert pooling in (Pooling.MEAN, Pooling.MAX)

        self.pooling = pooling
        self.nlp = nlp or spacy.load("en")

    def __call__(self, message: Dict, attribute: Text = "doc"):
        doc = message[attribute]
        if attribute == "text":
            doc = self.nlp(doc)

        features = np.array([t.vector for t in doc])
        cls_token_vec = self._calculate_cls_vector(features, self.pooling)
        features = np.concatenate([features, cls_token_vec])
        features = self._combine_with_existing_dense_features(message, features)
        return features

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

    @staticmethod
    def _combine_with_existing_dense_features(
        message: Dict, additional_features: Any, feature_name: Text = DENSE_FEATURES
    ) -> Any:
        if message.get(feature_name) is not None:

            if len(message.get(feature_name)) != len(additional_features):
                raise ValueError(
                    f"Cannot concatenate dense features as sequence dimension does not "
                    f"match: {len(message.get(feature_name))} != "
                    f"{len(additional_features)}. Message: '{message['text']}'."
                )

            return np.concatenate(
                (message.get(feature_name), additional_features), axis=-1
            )
        else:
            return additional_features
