# spacy_crfsuite: crfsuite entity extraction for spaCy.

`spacy_crfsuite` is an entity extraction pipeline for spaCy based .

## Install

**Python**

    pip install spacy_crfsuite

## Usage

spaCy usage

```python
import spacy

from spacy_crfsuite import CRFEntityExtractor

nlp = spacy.blank('en')
pipe = CRFEntityExtractor(nlp).from_disk("model.pkl")
nlp.add_pipe(pipe)

doc = nlp("show mexican restaurents up north")
for ent in doc.ents:
    print(ent.text, "--", ent.label_)
```

Train a model

```sh
python -m spacy_crfsuite.train <TRAIN> --model-dir <MODEL_DIR> --model-name <MODEL_NAME>
```

Evaluate a model

```sh
python -m spacy_crfsuite.eval <DEV> --model-dir <MODEL_DIR> --model-name <MODEL_NAME>
```

Gold annotations example (markdown)

```md
## Header
- what is my balance <!-- no entity -->
- how much do I have on my [savings](source_account) <!-- entity "source_account" has value "savings" -->
- how much do I have on my [savings account](source_account:savings) <!-- synonyms, method 1-->
- Could I pay in [yen](currency)?  <!-- entity matched by lookup table -->
```