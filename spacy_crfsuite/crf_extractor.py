import itertools
import joblib
import warnings
import sklearn_crfsuite

from collections import Counter
from pathlib import Path
from typing import Dict, Text, Any, Optional, List, Tuple, Union, Callable

from spacy.language import Language
from spacy.tokens.doc import Doc
from sklearn.metrics import classification_report, f1_score

from spacy_crfsuite.bilou import entity_name_from_tag, bilou_prefix_from_tag, NO_ENTITY_TAG
from spacy_crfsuite.compat import CRF
from spacy_crfsuite.features import CRFToken, Featurizer
from spacy_crfsuite.tokenizer import Token, SpacyTokenizer
from spacy_crfsuite.utils import override_defaults


class CRFExtractor:
    defaults: Dict[str, Any] = {
        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        "BILOU_flag": True,
        # crf_features is [before, token, after] array with before, token,
        # after holding keys about which features to use for each token,
        # for example, 'title' in array before will have the feature
        # "is the preceding token in title case?"
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
                "shape",
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

    function_dict: Dict[Text, Callable[[CRFToken], Any]] = {
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
        "pos2": lambda crf_token: crf_token.tag[:2] if crf_token.tag is not None else None,
        "upper": lambda crf_token: crf_token.text.isupper(),
        "digit": lambda crf_token: crf_token.text.isdigit(),
        "shape": lambda crf_token: crf_token.shape,
        "pattern": lambda crf_token: crf_token.pattern,
        "dense_features": lambda crf_token: crf_token.dense_features,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        ent_tagger: Optional["CRF"] = None,
    ) -> None:

        self.component_config = override_defaults(self.defaults, component_config)
        self.ent_tagger = ent_tagger
        self.featurizer = Featurizer(use_dense_features=self.use_dense_features())

    def from_disk(self, path: Union[Path, str] = "model.pkl") -> "CRFExtractor":
        """Load component from disk.

        Args:
            path (str, Path): path.

        Returns:
            CRFExtractor

        Raises:
            IOError
        """
        ent_tagger = joblib.load(path)
        assert isinstance(ent_tagger, sklearn_crfsuite.estimator.CRF)

        self.ent_tagger = ent_tagger
        return self

    def to_disk(self, path: Union[Path, str] = "model.pkl") -> None:
        """Save component to disk.

        Args:
            path (str, Path): path.

        Raises:
            RuntimeError - if entity tagger is not fitted for runtime.
        """
        self._check_runtime()
        joblib.dump(self.ent_tagger, path)

    def use_dense_features(self) -> bool:
        """A predicate to test if dense features should be used, according
        to the feature list decalred in component config.

        Returns:
            bool
        """
        for feature_list in self.component_config["features"]:
            if "dense_features" in feature_list:
                return True
        return False

    def process(self, example: Dict) -> List[Dict[Text, Any]]:
        """Process a single example with CRF entity tagging.

        Args:
            example (dict): example dict with either `doc`, `tokens` or `text` field.

        Returns:
            list.

        Raises:
            RuntimeError - in case crf tagger was not fitted.
        """
        self._check_runtime()

        crf_tokens: List[CRFToken] = self.featurizer(example)
        features: List[Dict[Text, Any]] = self._crf_tokens_to_features(crf_tokens)
        entities = self.ent_tagger.predict_marginals_single(features)
        return self._from_crf_to_json(example, entities)

    def train(
        self,
        training_samples: List[List[CRFToken]],
        dev_samples: Optional[List[List[CRFToken]]] = None,
    ) -> "CRFExtractor":
        """Train the entity tagger with examples.

        Args:
            training_samples (list): list of training examples.
            dev_samples (list): optional, list of dev examples.

        Returns:
            CRFExtractor
        """
        if self.ent_tagger is None:
            self.ent_tagger = CRF(
                algorithm=self.component_config["algorithm"],
                c1=self.component_config["c1"],
                c2=self.component_config["c2"],
                max_iterations=self.component_config["max_iter"],
                all_possible_transitions=self.component_config["all_possible_transitions"],
            )

        X_train = [self._crf_tokens_to_features(sent) for sent in training_samples]
        y_train = [self._crf_tokens_to_tags(sent) for sent in training_samples]

        if dev_samples:
            X_dev = [self._crf_tokens_to_features(sent) for sent in dev_samples]
            y_dev = [self._crf_tokens_to_tags(sent) for sent in dev_samples]
        else:
            X_dev = None
            y_dev = None

        self.ent_tagger.fit(X_train, y_train, X_dev=X_dev, y_dev=y_dev)
        return self

    def eval(self, eval_samples: List[List[CRFToken]]) -> Optional[Tuple[Any, Text]]:
        """Evaluate the entity tagger on dev examples.

        Args:
            eval_samples (list): list of dev examples.

        Returns:
            str, equivalent to output of ``sklearn.metrics.classification_report``
        """
        self._check_runtime()

        X_test = [self._crf_tokens_to_features(sent) for sent in eval_samples]
        y_test = [self._crf_tokens_to_tags(sent) for sent in eval_samples]
        y_test = list(itertools.chain.from_iterable(y_test))

        labels = list(self.ent_tagger.classes_)
        labels.remove(NO_ENTITY_TAG)
        sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))

        y_pred = self.ent_tagger.predict(X_test)
        y_pred = list(itertools.chain.from_iterable(y_pred))
        return classification_report(y_test, y_pred, labels=sorted_labels, digits=3)

    def fine_tune(
        self,
        val_samples: List[List[CRFToken]],
        cv=3,
        n_iter=10,
        n_jobs=-1,
        random_state=None,
    ):
        """Fine-tune hyper params (i.e L1/L2 regularization term).

        Args:
            val_samples (list): samples for cross-validation.
            cv (int): optional, cross-validation K-fold.
            n_iter (int): optional, num of iterations.
            n_jobs (int): optional, num of threads (-1 = ``multiprocessing.cpu_count()``).
            random_state (int): optional, random seed.

        Returns:
            a fitted instance of ``RandomizedSearchCV``.
        """
        import scipy

        from sklearn.metrics import make_scorer
        from sklearn.model_selection import RandomizedSearchCV

        params_space = {
            "c1": scipy.stats.expon(scale=0.5),
            "c2": scipy.stats.expon(scale=0.05),
        }
        crf = CRF(
            algorithm=self.component_config["algorithm"],
            max_iterations=self.component_config["max_iter"],
            all_possible_transitions=self.component_config["all_possible_transitions"],
        )

        X_train = [self._crf_tokens_to_features(sent) for sent in val_samples]
        y_train = [self._crf_tokens_to_tags(sent) for sent in val_samples]
        labels = list(set(itertools.chain.from_iterable(y_train)) - {NO_ENTITY_TAG})
        f1_scorer = make_scorer(f1_score, average="weighted", labels=labels)
        rs = RandomizedSearchCV(
            crf,
            params_space,
            cv=cv,
            verbose=1,
            n_jobs=n_jobs,
            n_iter=n_iter,
            scoring=f1_scorer,
            random_state=random_state,
        )
        rs.fit(X_train, y_train)
        return rs

    def _check_runtime(self):
        """Helper to check runtime before using component for predictions."""
        if self.ent_tagger is None:
            raise RuntimeError(
                "CRF tagger was not fitted. Make sure to call ``.train()`` "
                "to train a new model or ``.from_disk()`` to load "
                "a pre-trained model from disk."
            )

    def explain(self, n_trans=10, n_states=10) -> str:
        """Explain CRF learning by showing positive and negative examples of transitions and state features.

        See `sklearn-crfsuite` documentation for more details.

        ``https://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html#let-s-check-what-classifier-learned``

        Args:
            n_trans (int): num of transitions.
            n_states (int): num of state features.

        Returns:
            str
        """
        self._check_runtime()
        ret = ""

        trans_features = Counter(self.ent_tagger.transition_features_).most_common(n_trans)
        if trans_features:
            ret += "Most likely transitions:\n"
            ret += "\n".join(
                f"{label_from:10} -> {label_to:10} {weight:.6f}"
                for (label_from, label_to), weight in trans_features
            )

        trans_features = Counter(self.ent_tagger.transition_features_).most_common(
            n_trans * -1
        )
        if trans_features:
            ret += "\n\nMost unlikely transitions:\n"
            ret += "\n".join(
                f"{label_from:10} -> {label_to:10} {weight:.6f}"
                for (label_from, label_to), weight in trans_features
            )

        pos_features = Counter(self.ent_tagger.state_features_).most_common(n_states)
        if pos_features:
            ret += "\n\nPositive features:\n"
            ret += "\n".join(
                f"{weight:.6f} {label:10} {attr}" for (attr, label), weight in pos_features
            )

        neg_features = Counter(self.ent_tagger.state_features_).most_common(n_states * -1)
        if neg_features:
            ret += "\n\nNegative features:\n"
            ret += "\n".join(
                f"{weight:.6f} {label:10} {attr}" for (attr, label), weight in pos_features
            )

        return ret

    def most_likely_entity(self, idx: int, entities: List[Any]) -> Tuple[Text, Any]:
        if len(entities) > idx:
            entity_probs = entities[idx]
        else:
            entity_probs = None
        if entity_probs:
            label = max(entity_probs, key=lambda key: entity_probs[key])
            if self.component_config["BILOU_flag"]:
                # if we are using bilou flags, we will combine the prob
                # of the B, I, L and U tags for an entity (so if we have a
                # score of 60% for `B-address` and 40% and 30%
                # for `I-address`, we will return 70%)
                return (
                    label,
                    sum([v for k, v in entity_probs.items() if k[2:] == label[2:]]),
                )
            else:
                return label, entity_probs[label]
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
        return message.get("tokens")[:-1]

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
                warnings.warn(
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
                warnings.warn(
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

        if self.component_config["BILOU_flag"]:
            return self._convert_bilou_tagging_to_entity_result(message, tokens, entities)
        else:
            # not using BILOU tagging scheme, multi-word entities are split.
            return self._convert_simple_tagging_to_entity_result(tokens, entities)

    def _convert_bilou_tagging_to_entity_result(
        self, message: Dict, tokens: List[Token], entities: List[Dict[Text, float]]
    ) -> List[Dict[Text, Any]]:
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

    def _convert_simple_tagging_to_entity_result(
        self, tokens: List[Union[Token, Any]], entities: List[Any]
    ) -> List[Dict[Text, Any]]:
        json_ents = []
        for word_idx in range(len(tokens)):
            entity_label, confidence = self.most_likely_entity(word_idx, entities)
            word = tokens[word_idx]
            if entity_label != NO_ENTITY_TAG:
                ent = {
                    "start": word.start,
                    "end": word.end,
                    "value": word.text,
                    "entity": entity_label,
                    "confidence": confidence,
                }
                json_ents.append(ent)
        return json_ents

    def _crf_tokens_to_features(self, sentence: List[CRFToken]) -> List[Dict[Text, Any]]:
        """Convert the list of tokens into discrete features."""
        sentence_features = []
        n_tokens = len(sentence)
        for token_idx in range(n_tokens):
            token_features = self._features_for_token(sentence, token_idx)
            sentence_features.append(token_features)
        return sentence_features

    def _features_for_token(self, crf_tokens: List[CRFToken], token_idx: int):
        """Convert a token into discrete features including word before and word after."""
        token_features = {}
        configured_features = self.component_config["features"]
        window_size = len(configured_features)
        half_window_size = window_size // 2
        window_range = range(-half_window_size, half_window_size + 1)
        prefixes = [str(i) for i in window_range]

        for feature_idx in window_range:
            if token_idx + feature_idx >= len(crf_tokens):
                token_features["EOS"] = True
            elif token_idx + feature_idx < 0:
                token_features["BOS"] = True
            else:
                token = crf_tokens[token_idx + feature_idx]
                current_feature_index = feature_idx + half_window_size
                features = configured_features[current_feature_index]
                prefix = prefixes[current_feature_index]
                for feature in features:
                    if feature == "pattern":
                        # add all regexes extracted from the 'RegexFeaturizer' as a
                        # feature: 'pattern_name' is the name of the pattern the user
                        # set in the training data, 'matched' is either 'True' or
                        # 'False' depending on whether the token actually matches the
                        # pattern or not
                        regex_patterns = self.function_dict[feature](token)
                        for p_name, matched in regex_patterns.items():
                            feature_name = prefix + ":" + feature + ":" + p_name
                            token_features[feature_name] = matched
                    elif token and (feature == "pos" or feature == "pos2"):
                        value = self.function_dict[feature](token)
                        token_features[f"{prefix}:{feature}"] = value
                    else:
                        value = self.function_dict[feature](token)
                        token_features[prefix + ":" + feature] = value
        return token_features

    @staticmethod
    def _crf_tokens_to_tags(sentence: List[CRFToken]) -> List[Text]:
        return [crf_token.entity for crf_token in sentence]


class CRFEntityExtractor(object):
    """spaCy v2.0 pipeline component that sets entity annotations
    based on CRF (Conditional Random Field) estimator.


    See ```CRFExtractor``` for CRF implementation details.
    """

    name = "crf_ner"

    def __init__(self, nlp: Language, crf_extractor: Optional[CRFExtractor] = None):
        self.nlp = nlp
        self.crf_extractor = crf_extractor
        self.spacy_tokenizer = SpacyTokenizer(nlp)

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

        example = {"doc": doc, "text": doc.text}
        self.spacy_tokenizer.tokenize(example, attribute="doc")

        spans = [
            doc.char_span(
                entity_dict["start"], entity_dict["end"], label=entity_dict["entity"]
            )
            for entity_dict in self.crf_extractor.process(example)
        ]

        doc.ents = list(doc.ents) + spans
        with doc.retokenize() as retokenizer:
            for ent in doc.ents:
                retokenizer.merge(ent)

        return doc

    def from_disk(self, path: Union[Path, str], exclude=None) -> "CRFEntityExtractor":
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
