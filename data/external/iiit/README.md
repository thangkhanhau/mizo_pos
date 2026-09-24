# Mizo Universal Dependencies (UD) POS Dataset

## Overview
This dataset contains Part-of-Speech (POS) tagged text for the Mizo language, structured according to the Universal Dependencies (UD) guidelines. The data is formatted in CoNLL-U, making it ideal for training and evaluating POS taggers and related NLP models.

---

## CoNLL-U Format Description

Each line in a sentence contains 10 fields, separated by tabs:

| Column   | Description                                      | Example       |
|----------|--------------------------------------------------|---------------|
| ID       | Token index in the sentence (starting from 1)    | `1`           |
| FORM     | Word/token surface form                          | `lus`         |
| LEMMA    | Lemmatized/base form (may be `_` if unavailable) | `_`           |
| UPOS     | Universal POS tag (`NOUN`, `VERB`, etc.)         | `NOUN`        |
| XPOS     | Language-specific POS tag (optional)             | `_`           |
| FEATS    | Morphological features (optional)                | `_`           |
| HEAD     | Not applicable for POS tagging                   | `_`           |
| DEPREL   | Not applicable for POS tagging                   | `_`           |
| DEPS     | Not applicable for POS tagging                   | `_`           |
| MISC     | Miscellaneous info (optional)                    | `_`           |

---

## Dataset Statistics

| Split         | Sentences | Tokens | Types          |
|---------------|-----------|--------|----------------|
| Train         | 402       | 13,811 | 5,546          |
| Validation    | 50        | 1,835  | 712            |
| Test          | 50        | 1,611  | 654            |
| **Total**     | 502       | 17,257 | 5,429          |

---

## Usage
You can use this dataset for:
- Training POS taggers for Mizo.
- Linguistic research on Mizo grammar and syntax.
- Benchmarking performance of NLP models on low-resource languages.

---

## Main Contributors
- Soumyadip Ghosh  
- Nagaraju Vuppala  
- Henry Lalsiam
