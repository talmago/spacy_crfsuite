# spacy_crfsuite: CRF tagger for spaCy.

Sequence tagging with spaCy and crfsuite.

A port of [Rasa NLU](https://github.com/RasaHQ/rasa/blob/master/rasa/nlu/extractors/crf_entity_extractor.py).

## ✨ Features

- Simple but tough to beat **CRF entity tagger** (
  via [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite))
- **spaCy NER component**
- **Command line interface** for training & evaluation and **example notebook**
- [CoNLL](https://www.aclweb.org/anthology/W03-0419/), JSON
  and [Markdown](https://rasa.com/docs/rasa/nlu/training-data-format/#id5) **annotations**
- Pre-trained NER component

## ⏳ Installation

```bash
pip install spacy_crfsuite
```

## 🚀 Quickstart

### Usage as a spaCy pipeline component

```python
import spacy

from spacy.language import Language
from spacy_crfsuite import CRFEntityExtractor, CRFExtractor


@Language.factory("ner_crf")
def create_component(nlp, name):
    crf_extractor = CRFExtractor().from_disk("spacy_crfsuite_conll03_sm.bz2")
    return CRFEntityExtractor(nlp, crf_extractor=crf_extractor)


nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.add_pipe("ner_crf")

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

### Visualization (via [Gradio](https://gradio.app/named_entity_recognition/))

Run the command below to launch a Gradio playground

```sh
$ pip install gradio
$ python spacy_crfsuite/visualize.py
```

![](https://github.com/talmago/spacy_crfsuite/blob/master/img/gradio.png)


### Pre-trained models

You can download a pre-trained model.

| Dataset                                                                                               | F1  | 📥 Download                                                                                                                       |
|-------------------------------------------------------------------------------------------------------|-----|-----------------------------------------------------------------------------------------------------------------------------------|
| [CoNLL03](https://github.com/talmago/spacy_crfsuite/blob/master/examples/02%20-%20CoNLL%202003.ipynb) | 82% | [spacy_crfsuite_conll03_sm.bz2](https://github.com/talmago/spacy_crfsuite/releases/download/v1.1.0/spacy_crfsuite_conll03_sm.bz2) |

### Train your own model

Below is a command line to train a simple model for restaurants search bot with [markdown
annotations](https://github.com/talmago/spacy_crfsuite/blob/master/examples/restaurent_search.md) and save it to disk.
If you prefer working on jupyter, follow this [notebook](https://github.com/talmago/spacy_crfsuite/blob/master/examples/01%20-%20Custom%20Component.ipynb).


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

Below is a command line to test the CRF model and print the classification report (In the example we use the training set, however normally we would use a held out set).

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
ℹ Classification Report:
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

Now we can use the tagger for named entity recognition in a spaCy pipeline!

```python
import spacy

from spacy.language import Language
from spacy_crfsuite import CRFEntityExtractor, CRFExtractor


@Language.factory("ner_crf")
def create_component(nlp, name):
    crf_extractor = CRFExtractor().from_disk("model/model.pkl")
    return CRFEntityExtractor(nlp, crf_extractor=crf_extractor)


nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.add_pipe("ner_crf")

doc = nlp("show mexican restaurents up north")
for ent in doc.ents:
    print(ent.text, "--", ent.label_)

# Output:
# mexican -- cuisine
# north -- location
```

Or alternatively as a standalone component

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

> **Notice**: You can also access the `crf_extractor` directly with ```nlp.get_pipe("crf_ner").crf_extractor```.

### Deploy to a web server

Start a web service

```sh
$ pip install uvicorn
$ uvicorn spacy_crfsuite.serve:app --host 127.0.0.1 --port 5000
```

>Notice: Set `$SPACY_MODEL` and `$CRF_MODEL` in your environment to control the server configurations

cURL example

```sh
$ curl -X POST http://127.0.0.1:5000/parse -H 'Content-Type: application/json' -d '{"text": "George Walker Bush (born July 6, 1946) is an American politician and businessman who served as the 43rd president of the United States from 2001 to 2009."}'
{
  "data": [
    {
      "text": "George Walker Bush (born July 6, 1946) is an American politician and businessman who served as the 43rd president of the United States from 2001 to 2009.",
      "entities": [
        {
          "start": 0,
          "end": 18,
          "value": "George Walker Bush",
          "entity": "PER"
        },
        {
          "start": 45,
          "end": 53,
          "value": "American",
          "entity": "MISC"
        },
        {
          "start": 121,
          "end": 134,
          "value": "United States",
          "entity": "LOC"
        }
      ]
    }
  ]
}
```

## Development

Set up env

```sh
$ poetry install
$ poetry run spacy download en_core_web_sm
```

Run unit test

```sh
$ poetry run pytest
```

Run black (code formatting)

```sh
$ poetry run black spacy_crfsuite/ --config=pyproject.toml
```
