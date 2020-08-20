import logging
from pathlib import Path
from typing import Dict, Text, Any, Optional, List, Tuple, NamedTuple, Union

import joblib
import numpy as np
import sklearn_crfsuite
from sklearn_crfsuite import CRF, metrics
from spacy.language import Language
from spacy.tokens.doc import Doc

from spacy_crfsuite.bilou import (
    entity_name_from_tag,
    bilou_tags_from_offsets,
    bilou_prefix_from_tag,
)
from spacy_crfsuite.constants import TOKENS, PATTERN, DENSE_FEATURES
from spacy_crfsuite.dense_features import DenseFeatures
from spacy_crfsuite.tokenizer import Token, SpacyTokenizer
from spacy_crfsuite.utils import override_defaults

LOG = logging.getLogger("crf")


class CRFToken(NamedTuple):
    text: Text
    tag: Text
    entity: Text
    pattern: Dict[Text, Any]
    dense_features: np.ndarray


class CRFExtractor:
    defaults = {
        # crf_features is [before, token, after] array with before, token,
        # after holding keys about which features to use for each token,
        # for example, 'title' in array before will have the feature
        # "is the preceding token in title case?"
        # POS features require SpacyTokenizer
        # pattern feature require RegexFeaturizer
        "features": [
            ["low", "title", "upper"],
            [
                "low",
                "bias",
                "prefix5",
                "prefix2",
                "suffix5",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
                "pattern",
            ],
            ["low", "title", "upper"],
        ],
        # algorithm
        "algorithm": "lbfgs",
        # The maximum number of iterations for optimization algorithms.
        "max_iter": 1000,
        # weight of the L1 regularization
        "c1": 0.1,
        # weight of the L2 regularization
        "c2": 0.1,
        # CRF transition
        "all_possible_transitions": True,
    }

    function_dict = {
        "low": lambda crf_token: crf_token.text.lower(),
        "title": lambda crf_token: crf_token.text.istitle(),
        "prefix5": lambda crf_token: crf_token.text[:5],
        "prefix2": lambda crf_token: crf_token.text[:2],
        "suffix5": lambda crf_token: crf_token.text[-5:],
        "suffix3": lambda crf_token: crf_token.text[-3:],
        "suffix2": lambda crf_token: crf_token.text[-2:],
        "suffix1": lambda crf_token: crf_token.text[-1:],
        "bias": lambda crf_token: "bias",
        "pos": lambda crf_token: crf_token.tag,
        "pos2": lambda crf_token: crf_token.tag[:2]
        if crf_token.tag is not None
        else None,
        "upper": lambda crf_token: crf_token.text.isupper(),
        "digit": lambda crf_token: crf_token.text.isdigit(),
        "pattern": lambda crf_token: crf_token.pattern
        if crf_token.pattern is not None
        else None,
        "text_dense_features": lambda crf_token: crf_token.dense_features,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        ent_tagger: Optional["CRF"] = None,
    ) -> None:

        self.component_config = override_defaults(self.defaults, component_config)
        self.ent_tagger = ent_tagger

    def from_disk(self, path: Union[Path, str] = "model.pkl") -> "CRFExtractor":
        self.ent_tagger = joblib.load(path)
        return self

    def to_disk(self, path: Union[Path, str] = "model.pkl") -> None:
        """Save model to disk."""
        if not self.ent_tagger:
            return

        joblib.dump(self.ent_tagger, path)

    def train(self, training_samples: List[List[CRFToken]]) -> "CRFExtractor":
        """Train the crf tagger based on the training data."""
        if self.ent_tagger is None:
            self.ent_tagger = sklearn_crfsuite.CRF(
                algorithm=self.component_config["algorithm"],
                c1=self.component_config["c1"],
                c2=self.component_config["c2"],
                max_iterations=self.component_config["max_iter"],
                all_possible_transitions=self.component_config[
                    "all_possible_transitions"
                ],
            )
        X_train = [self._sentence_to_features(sent) for sent in training_samples]
        y_train = [self._sentence_to_labels(sent) for sent in training_samples]
        self.ent_tagger.fit(X_train, y_train)
        return self

    def eval(self, eval_samples: List[List[CRFToken]]) -> Optional[Tuple[Any, Text]]:
        """Train the crf tagger based on the training data."""
        if self.ent_tagger is None:
            raise RuntimeError(".eval() was called before .train() ?")

        X_test = [self._sentence_to_features(sent) for sent in eval_samples]
        y_test = [self._sentence_to_labels(sent) for sent in eval_samples]

        labels = list(self.ent_tagger.classes_)
        labels.remove("O")
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

        y_pred = self.ent_tagger.predict(X_test)
        f1_score = metrics.flat_f1_score(
            y_test, y_pred, average="weighted", labels=labels
        )
        classification_report = metrics.flat_classification_report(
            y_test, y_pred, labels=sorted_labels, digits=3
        )
        return f1_score, classification_report

    def process(self, message: Dict) -> List[Dict[Text, Any]]:
        """Take a sentence and return entities in json format"""
        if self.ent_tagger is not None:
            text_data = self._from_text_to_crf(message)
            features = self._sentence_to_features(text_data)
            entities = self.ent_tagger.predict_marginals_single(features)
            return self._from_crf_to_json(message, entities)
        else:
            return []

    def use_dense_features(self) -> bool:
        for feature_list in self.component_config["features"]:
            if DENSE_FEATURES in feature_list:
                return True
        return False

    def most_likely_entity(self, idx: int, entities: List[Any]) -> Tuple[Text, Any]:
        if len(entities) > idx:
            entity_probs = entities[idx]
        else:
            entity_probs = None
        if entity_probs:
            label = max(entity_probs, key=lambda key: entity_probs[key])
            # if we are using bilou flags, we will combine the prob
            # of the B, I, L and U tags for an entity (so if we have a
            # score of 60% for `B-address` and 40% and 30%
            # for `I-address`, we will return 70%)
            return (
                label,
                sum([v for k, v in entity_probs.items() if k[2:] == label[2:]]),
            )
        else:
            return "", 0.0

    @staticmethod
    def _create_entity_dict(
        message: Dict,
        tokens: List[Token],
        start: int,
        end: int,
        entity: str,
        confidence: float,
    ) -> Dict[Text, Any]:

        _start = tokens[start].start
        _end = tokens[end].end
        value = tokens[start].text
        value += "".join(
            [
                message["text"][tokens[i - 1].end : tokens[i].start] + tokens[i].text
                for i in range(start + 1, end + 1)
            ]
        )

        return {
            "start": _start,
            "end": _end,
            "value": value,
            "entity": entity,
            "confidence": confidence,
        }

    @staticmethod
    def _tokens_without_cls(message: Dict) -> List[Token]:
        # [:-1] to remove the CLS token from the list of tokens
        return message.get(TOKENS)[:-1]

    def _find_bilou_end(self, word_idx, entities) -> Any:
        ent_word_idx = word_idx + 1
        finished = False

        # get information about the first word, tagged with `B-...`
        label, confidence = self.most_likely_entity(word_idx, entities)
        entity_label = entity_name_from_tag(label)

        while not finished:
            label, label_confidence = self.most_likely_entity(ent_word_idx, entities)

            confidence = min(confidence, label_confidence)

            if label[2:] != entity_label:
                # words are not tagged the same entity class
                LOG.debug(
                    "Inconsistent BILOU tagging found, B- tag, L- "
                    "tag pair encloses multiple entity classes.i.e. "
                    "[B-a, I-b, L-a] instead of [B-a, I-a, L-a].\n"
                    "Assuming B- class is correct."
                )

            if label.startswith("L-"):
                # end of the entity
                finished = True
            elif label.startswith("I-"):
                # middle part of the entity
                ent_word_idx += 1
            else:
                # entity not closed by an L- tag
                finished = True
                ent_word_idx -= 1
                LOG.debug(
                    "Inconsistent BILOU tagging found, B- tag not "
                    "closed by L- tag, i.e [B-a, I-a, O] instead of "
                    "[B-a, L-a, O].\nAssuming last tag is L-"
                )
        return ent_word_idx, confidence

    def _handle_bilou_label(
        self, word_idx: int, entities: List[Any]
    ) -> Tuple[Any, Any, Any]:
        label, confidence = self.most_likely_entity(word_idx, entities)
        entity_label = entity_name_from_tag(label)

        if bilou_prefix_from_tag(label) == "U":
            return word_idx, confidence, entity_label

        elif bilou_prefix_from_tag(label) == "B":
            # start of multi word-entity need to represent whole extent
            ent_word_idx, confidence = self._find_bilou_end(word_idx, entities)
            return ent_word_idx, confidence, entity_label

        else:
            return None, None, None

    def _from_crf_to_json(
        self, message: Dict, entities: List[Any]
    ) -> List[Dict[Text, Any]]:
        tokens = self._tokens_without_cls(message)
        if len(tokens) != len(entities):
            raise Exception(
                "Inconsistency in amount of tokens between crfsuite and message"
            )

        return self._convert_bilou_tagging_to_entity_result(message, tokens, entities)

    def _convert_bilou_tagging_to_entity_result(
        self, message: Dict, tokens: List[Token], entities: List[Dict[Text, float]]
    ):
        # using the BILOU tagging scheme
        json_ents = []
        word_idx = 0
        while word_idx < len(tokens):
            end_idx, confidence, entity_label = self._handle_bilou_label(
                word_idx, entities
            )
            if end_idx is not None:
                ent = self._create_entity_dict(
                    message, tokens, word_idx, end_idx, entity_label, confidence
                )
                json_ents.append(ent)
                word_idx = end_idx + 1
            else:
                word_idx += 1
        return json_ents

    def _sentence_to_features(self, sentence: List[CRFToken]) -> List[Dict[Text, Any]]:
        """Convert a word into discrete features in self.crf_features,
        including word before and word after."""

        configured_features = self.component_config["features"]
        sentence_features = []

        for word_idx in range(len(sentence)):
            # word before(-1), current word(0), next word(+1)
            feature_span = len(configured_features)
            half_span = feature_span // 2
            feature_range = range(-half_span, half_span + 1)
            prefixes = [str(i) for i in feature_range]
            word_features = {}
            for f_i in feature_range:
                if word_idx + f_i >= len(sentence):
                    word_features["EOS"] = True
                    # End Of Sentence
                elif word_idx + f_i < 0:
                    word_features["BOS"] = True
                    # Beginning Of Sentence
                else:
                    word = sentence[word_idx + f_i]
                    f_i_from_zero = f_i + half_span
                    prefix = prefixes[f_i_from_zero]
                    features = configured_features[f_i_from_zero]
                    for feature in features:
                        if feature == "pattern":
                            # add all regexes as a feature
                            regex_patterns = self.function_dict[feature](word)
                            # pytype: disable=attribute-error
                            for p_name, matched in regex_patterns.items():
                                feature_name = prefix + ":" + feature + ":" + p_name
                                word_features[feature_name] = matched
                            # pytype: enable=attribute-error
                        elif word and (feature == "pos" or feature == "pos2"):
                            value = self.function_dict[feature](word)
                            word_features[f"{prefix}:{feature}"] = value
                        else:
                            # append each feature to a feature vector
                            value = self.function_dict[feature](word)
                            word_features[prefix + ":" + feature] = value

            sentence_features.append(word_features)
        return sentence_features

    @staticmethod
    def _sentence_to_labels(
        sentence: List[
            Tuple[
                Optional[Text],
                Optional[Text],
                Text,
                Dict[Text, Any],
                Optional[Dict[str, Any]],
            ]
        ],
    ) -> List[Text]:

        return [label for _, _, label, _, _ in sentence]

    def from_json_to_crf(
        self, message: Dict, entity_offsets: List[Tuple[int, int, Text]]
    ) -> List[CRFToken]:
        """Convert json examples to format of underlying crfsuite."""

        tokens = self._tokens_without_cls(message)
        ents = bilou_tags_from_offsets(tokens, entity_offsets)

        # collect badly annotated examples
        collected = []
        for t, e in zip(tokens, ents):
            if e == "-":
                collected.append(t)
            elif collected:
                collected_text = " ".join([t.text for t in collected])
                LOG.warning(
                    f"Misaligned entity annotation for '{collected_text}' "
                    f"in sentence '{message.text}' with intent "
                    f"'{message.get('intent')}'. "
                    f"Make sure the start and end values of the "
                    f"annotated training examples end at token "
                    f"boundaries (e.g. don't include trailing "
                    f"whitespaces or punctuation)."
                )
                collected = []

        return self._from_text_to_crf(message, ents)

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
            LOG.warning(
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
                str(index): token_features
                for index, token_features in enumerate(feature)
            }
            converted = {DENSE_FEATURES: feature_dict}
            features_out.append(converted)
        return features_out

    def _from_text_to_crf(
        self, message: Dict, entities: List[Text] = None
    ) -> List[CRFToken]:
        """Takes a sentence and switches it to crfsuite format."""
        crf_format = []
        tokens = self._tokens_without_cls(message)
        text_dense_features = self.__get_dense_features(message)
        for i, token in enumerate(tokens):
            pattern = self.__pattern_of_token(message, i)
            entity = entities[i] if entities else "N/A"
            tag = token.get("pos")
            dense_features = (
                text_dense_features[i] if text_dense_features is not None else []
            )
            crf_format.append(CRFToken(token.text, tag, entity, pattern, dense_features))
        return crf_format


class CRFEntityExtractor(object):
    """spaCy v2.0 pipeline component that sets entity annotations
    based on CRF (Conditional Random Field) estimator.


    See ```CRFExtractor``` for CRF implementation details.
    """

    name = "crf_entity_extractor"

    def __init__(self, nlp: Language, crf_extractor: Optional[CRFExtractor] = None):
        self.nlp = nlp
        self.crf_extractor = crf_extractor

    def __call__(self, doc: Doc):
        """Apply the pipeline component on a Doc object and modify it if matches
        are found. Return the Doc, so it can be processed by the next component
        in the pipeline, if available.

        References:
            - ``https://spacy.io/usage/processing-pipelines#component-example2``.

        Args:
            doc (Doc): spaCy document.

        Returns:
            doc
        """
        if not self.crf_extractor:
            raise RuntimeError(
                "`CRFEntityExtractor` was not initialized. "
                "Did you call `.from_disk()` method ?"
            )

        tokenizer = SpacyTokenizer(self.nlp)
        message = {"doc": doc, "text": doc.text}

        tokens = tokenizer.tokenize(message, attribute="doc")
        tokenizer.add_cls_token(tokens)
        message["tokens"] = tokens

        if self.crf_extractor.use_dense_features():
            dense_features = DenseFeatures(self.nlp)
            text_dense_features = dense_features(message, attribute="doc")
            if len(text_dense_features) > 0:
                message["text_dense_features"] = text_dense_features

        spans = [
            doc.char_span(
                entity_dict["start"], entity_dict["end"], label=entity_dict["entity"]
            )
            for entity_dict in self.crf_extractor.process(message)
        ]

        doc.ents = list(doc.ents) + spans
        for span in spans:
            # Iterate over all spans and merge them into one token. This is done
            # after setting the entities – otherwise, it would cause mismatched
            # indices!
            span.merge()

        return doc

    def from_disk(self, path: Union[Path, str]) -> "CRFEntityExtractor":
        """Load crf extractor from disk.

        Args:
            path: path to directory.

        Returns:
            Component
        """
        if not isinstance(path, Path):
            path = Path(path)

        self.crf_extractor = CRFExtractor().from_disk(path)
        return self
