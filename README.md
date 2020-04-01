# spacy_crfsuite: crfsuite entity extraction for spaCy.

`spacy_crfsuite` is an entity extraction pipeline for spaCy based .

## Install

**Python**

    pip install spacy_crfsuite

## Usage

Spacy usage

```python
import os
import spacy

from spacy_crfsuite import CRFEntityExtractorFactory

# load spacy language model
nlp = spacy.blank('en')

# Will look for ``crf.pkl`` in current working dir
pipe = CRFEntityExtractorFactory(nlp, model_dir=os.getcwd())
nlp.add_pipe(pipe)

# Use CRF to extract entities
doc = nlp("given we launched L&M a couple of years ago")
for ent in doc.ents:
    print(ent.text, "--", ent.label_)
```

Train a model

```sh
python -m spacy_crfsuite.trainer train <TRAIN> --model-dir <MODEL_DIR> --model-name <MODEL_NAME>
```

Evaluate a model

```sh
python -m spacy_crfsuite.trainer eval <DEV> --model-dir <MODEL_DIR> --model-name <MODEL_NAME>
```

Gold annotations example (markdown)

```md
## Header
- what is my balance <!-- no entity -->
- how much do I have on my [savings](source_account) <!-- entity "source_account" has value "savings" -->
- how much do I have on my [savings account](source_account:savings) <!-- synonyms, method 1-->
- Could I pay in [yen](currency)?  <!-- entity matched by lookup table -->
```