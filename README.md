# spacy_crfsuite: CRF entity tagger for spaCy.

## ✨ Features

- **spaCy NER component** for **Conditional Random Field** entity extraction (via [sklearn-crfsuite](https://github.com/TeamHG-Memex/sklearn-crfsuite)).
- train & eval command line and example notebook.
- supports **JSON, CoNLL and [Markdown annotations](https://rasa.com/docs/rasa/nlu/training-data-format/#id5)** 

## Installation

**Python**

    pip install spacy_crfsuite

## 🚀 Quickstart

### Usage as a spaCy pipeline component

spaCy pipeline

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

Follow this example [notebook](https://github.com/talmago/spacy_crfsuite/blob/master/examples/example.ipynb) 
to train the CRF entity tagger from few [restaurant search examples](https://github.com/talmago/spacy_crfsuite/blob/master/examples/example.md).


## Train & evaluate CRF entity tagger

Set up configuration file

```sh
$ cat << EOF > config.json
{"c1": 0.03, "c2": 0.06}
EOF
```

Run training

```sh
$ python -m spacy_crfsuite.train examples/example.md -o model/ -c config.json
ℹ Loading config: config.json
ℹ Training CRF entity tagger with 15 examples.
ℹ Saving model to disk
✔ Successfully saved model to file.
/Users/talmago/git/spacy_crfsuite/model/model.pkl
```

Evaluate on a dataset

```sh
$ python -m spacy_crfsuite.eval examples/example.md -m model/model.pkl
ℹ Loading model from file
model/model.pkl
✔ Successfully loaded CRF tagger
<spacy_crfsuite.crf_extractor.CRFExtractor object at 0x126e5f438>
ℹ Loading dev dataset from file
examples/example.md
✔ Successfully loaded 15 dev examples.
⚠ f1 score: 1.0
              precision    recall  f1-score   support

           -      1.000     1.000     1.000         2
   B-cuisine      1.000     1.000     1.000         1
   L-cuisine      1.000     1.000     1.000         1
   U-cuisine      1.000     1.000     1.000         5
  U-location      1.000     1.000     1.000         2

   micro avg      1.000     1.000     1.000        11
   macro avg      1.000     1.000     1.000        11
weighted avg      1.000     1.000     1.000        11
```