import spacy

from abc import ABCMeta, abstractmethod
from typing import Text, Optional, Any, Dict, List

from spacy_crfsuite.constants import CLS_TOKEN


class Token:
    def __init__(
        self,
        text: Text,
        start: int,
        end: Optional[int] = None,
        data: Optional[Dict[Text, Any]] = None,
        lemma: Optional[Text] = None,
    ) -> None:
        self.text = text
        self.start = start
        self.end = end if end else start + len(text)

        self.data = data if data else {}
        self.lemma = lemma or text

    def set(self, prop: Text, info: Any) -> None:
        self.data[prop] = info

    def get(self, prop: Text, default: Optional[Any] = None) -> Any:
        return self.data.get(prop, default)

    def __eq__(self, other):
        if not isinstance(other, Token):
            return NotImplemented
        return (self.start, self.end, self.text, self.lemma) == (
            other.start,
            other.end,
            other.text,
            other.lemma,
        )

    def __lt__(self, other):
        if not isinstance(other, Token):
            return NotImplemented

        return (self.start, self.end, self.text, self.lemma) < (
            other.start,
            other.end,
            other.text,
            other.lemma,
        )


class Tokenizer:
    __metaclass__ = ABCMeta

    @abstractmethod
    def tokenize(self, message: Dict, attribute: Text = "text") -> List[Token]:
        raise NotImplementedError("should be implemented by subclass")


class SpacyTokenizer(Tokenizer):
    def __init__(self, nlp=None):
        self.nlp = nlp or spacy.blank("en")

    def tokenize(self, message: Dict, attribute: Text = "text") -> List[Token]:
        doc = message[attribute]

        if attribute == "text":
            doc = self.nlp(doc)

        return [
            Token(t.text, t.idx, lemma=t.lemma_, data={"pos": self._tag_of_token(t)})
            for t in doc
        ]

    @staticmethod
    def _tag_of_token(token: Any) -> Text:
        import spacy

        if spacy.about.__version__ > "2" and token._.has("tag"):
            return token._.get("tag")
        else:
            return token.tag_

    @staticmethod
    def add_cls_token(tokens: List[Token]) -> None:
        # +1 to have a space between the last token and the __cls__ token
        idx = tokens[-1].end + 1
        tokens.append(Token(CLS_TOKEN, idx))
