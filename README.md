# spacy_crfsuite: CRF entity tagger for spaCy.

## ✨ Features

- Simple but tough to beat **CRF entity tagger** (via [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite))
- **spaCy NER component**
- **Command line interface** for training & evaluation command line and **example notebook**
- **[CoNLL](https://www.aclweb.org/anthology/W03-0419/), JSON and [Markdown](https://rasa.com/docs/rasa/nlu/training-data-format/#id5) annotations** 

## Installation

**Python**

    pip install spacy_crfsuite

## 🚀 Quickstart

### Standalone usage

```python
from spacy_crfsuite import CRFExtractor, prepare_example

crf_extractor = CRFExtractor().from_disk("model.pkl")
raw_example = {"text": "show mexican restaurents up north"}
example = prepare_example(raw_example, crf_extractor=crf_extractor)
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

### Usage as a spaCy pipeline component

```python
import spacy

from spacy_crfsuite import CRFEntityExtractor

nlp = spacy.blank('en')
pipe = CRFEntityExtractor(nlp).from_disk("model.pkl")
nlp.add_pipe(pipe)

doc = nlp("show mexican restaurents up north")
for ent in doc.ents:
    print(ent.text, "--", ent.label_)

# Output:
# mexican -- cuisine
# north -- location
```

Follow this [notebook](https://github.com/talmago/spacy_crfsuite/blob/master/examples/01%20-%20Custom%20Component.ipynb) 
to learn how to train a entity tagger from few [restaurant search examples](https://github.com/talmago/spacy_crfsuite/blob/master/examples/data/example.md).


### Train & evaluate custom CRF tagger

Set up configuration file

```sh
$ cat << EOF > config.json
{"c1": 0.03, "c2": 0.06}
EOF
```

Run training

```sh
$ python -m spacy_crfsuite.train examples/data/example.md -o model/ -c config.json
ℹ Loading config: config.json
ℹ Training CRF entity tagger with 15 examples.
ℹ Saving model to disk
✔ Successfully saved model to file.
/Users/talmago/git/spacy_crfsuite/model/model.pkl
```

Evaluate on a dataset

```sh
$ python -m spacy_crfsuite.eval examples/data/example.md -m model/model.pkl
ℹ Loading model from file
model/model.pkl
✔ Successfully loaded CRF tagger
<spacy_crfsuite.crf_extractor.CRFExtractor object at 0x126e5f438>
ℹ Loading dev dataset from file
examples/example.md
✔ Successfully loaded 15 dev examples.
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

Explain model

```python
print(crf_extractor.explain())

# Output:
#
# Most likely transitions:
# O          -> O          1.617362
# U-cuisine  -> O          1.277659
# B-cuisine  -> I-cuisine  1.206597
# I-cuisine  -> L-cuisine  0.800963
# O          -> U-location 0.719703
# B-cuisine  -> L-cuisine  0.589600
# O          -> U-cuisine  0.402591
# U-location -> U-cuisine  0.325804
# O          -> B-cuisine  0.150878
# L-cuisine  -> O          0.087336
# 
# Positive features:
# 2.186071 O          0:bias:bias
# 1.973212 U-location -1:low:the
# 1.135395 B-cuisine  -1:low:for
# 1.121395 U-location 0:prefix5:centr
# 1.121395 U-location 0:prefix2:ce
# 1.106081 U-location 0:digit
# 1.019241 U-cuisine  0:prefix5:chine
# 1.019241 U-cuisine  0:prefix2:ch
# 1.011240 U-cuisine  0:suffix2:an
# 0.945071 U-cuisine  -1:low:me
```

## Development

Set up pip & virtualenv

```sh
$ pipenv sync -d
```

Run unit test

```sh
$ pipenv run pytest
```

Run black (code formatter)

```sh
$ pipenv run black spacy_crfsuite/ --config=pyproject.toml
```