# spacy_crfsuite: CRF tagger for spaCy.

Sequence tagging with spaCy and crfsuite.

Copied from [Rasa NLU](https://github.com/RasaHQ/rasa/blob/master/rasa/nlu/extractors/crf_entity_extractor.py).

## ✨ Features

- Simple but tough to beat **CRF entity tagger** (via [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite))
- **spaCy NER component**
- **Command line interface** for training & evaluation and **example notebook**
- [CoNLL](https://www.aclweb.org/anthology/W03-0419/), JSON and [Markdown](https://rasa.com/docs/rasa/nlu/training-data-format/#id5) **annotations**
- Pre-trained NER component 

## ⏳ Installation

```bash
pip install spacy_crfsuite
```

## 🚀 Quickstart

### Usage as a spaCy pipeline component

```python
import spacy

from spacy_crfsuite import CRFEntityExtractor, CRFExtractor

@Language.factory("ner-crf")
def create_my_component(nlp, name):
    crf_extractor = CRFExtractor().from_disk("spacy_crfsuite_conll03_sm.bz2")
    return CRFEntityExtractor(nlp, crf_extractor=crf_extractor)


nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.add_pipe("ner-crf")

doc = nlp(
    "George Walker Bush (born July 6, 1946) is an American politician and businessman "
    "who served as the 43rd president of the United States from 2001 to 2009.")

for ent in doc.ents:
    print(ent, "-", ent.label_)

# Output:
# George Walker Bush - PER
# American - MISC
# United States - LOC
```

### Pre-trained models

You can download a pre-trained model.

| Dataset              |  F1   | 📥 Download                                                                                                                                                                                                                                                                                                   |
| -------------------- | ------  | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| [CoNLL03](https://github.com/talmago/spacy_crfsuite/blob/master/examples/02%20-%20CoNLL%202003.ipynb)            |   82% | [spacy_crfsuite_conll03_sm.bz2](https://github.com/talmago/spacy_crfsuite/releases/download/v1.1.0/spacy_crfsuite_conll03_sm.bz2) |


### Train your own model

Let's train a simple model for restaurent search bot with [markdown 
annotations](https://github.com/talmago/spacy_crfsuite/blob/master/examples/restaurent_search.md) and the command line. 
You can also try this [notebook](https://github.com/talmago/spacy_crfsuite/blob/master/examples/01%20-%20Custom%20Component.ipynb).

So we start by training a model and saving it to disk.

```sh
$ python -m spacy_crfsuite.train examples/restaurent_search.md -c examples/default-config.json -o model/ -lm en_core_web_sm
ℹ Loading config from disk
✔ Successfully loaded config from file.
examples/default-config.json
ℹ Loading training examples.
✔ Successfully loaded 15 training examples from file.
examples/restaurent_search.md
ℹ Using spaCy model: en_core_web_sm
ℹ Training entity tagger with CRF.
ℹ Saving model to disk
✔ Successfully saved model to file.
model/model.pkl
```

We can also evaluate on a dev set to get f1 & classification report. Below we use the training examples.

```sh
$ python -m spacy_crfsuite.eval examples/restaurent_search.md -m model/model.pkl -lm en_core_web_sm
ℹ Loading model from file
model/model.pkl
✔ Successfully loaded CRF tagger
<spacy_crfsuite.crf_extractor.CRFExtractor object at 0x126e5f438>
ℹ Loading dev dataset from file
examples/example.md
✔ Successfully loaded 15 dev examples.
ℹ Using spaCy model: en_core_web_sm
⚠ f1 score: 1.0
              precision    recall  f1-score   support

   B-cuisine      1.000     1.000     1.000         2
   I-cuisine      1.000     1.000     1.000         1
   L-cuisine      1.000     1.000     1.000         2
   U-cuisine      1.000     1.000     1.000         5
  U-location      1.000     1.000     1.000         7

   micro avg      1.000     1.000     1.000        17
   macro avg      1.000     1.000     1.000        17
weighted avg      1.000     1.000     1.000        17
```

Now we can use the tagger in a spaCy pipeline!

```python
import spacy

from spacy_crfsuite import CRFEntityExtractor

nlp = spacy.load('en_core_web_sm')
pipe = CRFEntityExtractor(nlp).from_disk("model/model.pkl")
nlp.add_pipe(pipe)

doc = nlp("show mexican restaurents up north")
for ent in doc.ents:
    print(ent.text, "--", ent.label_)

# Output:
# mexican -- cuisine
# north -- location
```

Or alternatively as a standalone component.

```python
from spacy_crfsuite import CRFExtractor
from spacy_crfsuite.tokenizer import SpacyTokenizer

crf_extractor = CRFExtractor().from_disk("model/model.pkl")
tokenizer = SpacyTokenizer()

example = {"text": "show mexican restaurents up north"}
tokenizer.tokenize(example, attribute="text")
crf_extractor.process(example)

# Output:
# [{'start': 5,
#   'end': 12,
#   'value': 'mexican',
#   'entity': 'cuisine',
#   'confidence': 0.5823148506311286},
#  {'start': 28,
#   'end': 33,
#   'value': 'north',
#   'entity': 'location',
#   'confidence': 0.8863076478494413}]
```

We can also take a look at what model learned.

Use the `.explain()` method to understand model decision.

```python
print(crf_extractor.explain())

# Output:
#
# Most likely transitions:
# O          -> O          1.637338
# B-cuisine  -> I-cuisine  1.373766
# U-cuisine  -> O          1.306077
# I-cuisine  -> L-cuisine  0.915989
# O          -> U-location 0.751463
# B-cuisine  -> L-cuisine  0.698893
# O          -> U-cuisine  0.480360
# U-location -> U-cuisine  0.403487
# O          -> B-cuisine  0.261450
# L-cuisine  -> O          0.182695
# 
# Positive features:
# 1.976502 O          0:bias:bias
# 1.957180 U-location -1:low:the
# 1.216547 B-cuisine  -1:low:for
# 1.153924 U-location 0:prefix5:centr
# 1.153924 U-location 0:prefix2:ce
# 1.110536 U-location 0:digit
# 1.058294 U-cuisine  0:prefix5:chine
# 1.058294 U-cuisine  0:prefix2:ch
# 1.051457 U-cuisine  0:suffix2:an
# 0.999976 U-cuisine  -1:low:me
```

>**Notice**: You can also access the `crf_extractor` directly with ```nlp.get_pipe("crf_ner").crf_extractor```.

## Development

Set up virtualenv

```sh
$ pipenv sync -d
```

Run unit test

```sh
$ pipenv run pytest
```

Run black (code formatting)

```sh
$ pipenv run black spacy_crfsuite/ --config=pyproject.toml
```