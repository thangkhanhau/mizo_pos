# mizo_pos: Silver-Standard PoS Corpus and Neural Tagger for Mizo

Code, trained tagger, lexicon, correction rules and a corpus sample for the paper:

> Haulai, T., Hussain, J., Dawngliani, M. S., and Vargas-Alejo, V.
> *Projection and Correction: Building a Silver-Standard Part-of-Speech Corpus
> and Neural Tagger for Mizo.* Submitted to Language Resources and Evaluation.

The pipeline projects English Universal Dependencies (UD) tags onto Mizo
through SimAlign word alignments, verifies them against a Mizo PoS lexicon,
corrects them through five rounds of native-speaker review, and trains a
character-aware BiLSTM-CRF tagger.

## Results

| System | Silver test accuracy | Gold (Ghosh et al., 2025), scheme-neutral |
|---|---|---|
| Most-frequent-tag lexicon | 93.10% | - |
| BiLSTM-CRF | 94.49% | 55.69% |

Silver figures measure consistency with the pipeline's own labels, not
annotation quality. See Sections 5.6 and 7 of the paper.

## Contents

| Path | Contents |
|---|---|
| `src/model.py` | BiLSTM-CRF model, loading and tagging helpers, scoring |
| `src/lexicon_baseline.py` | Most-frequent-tag baseline |
| `src/eval_external_gold.py` | Evaluation on external gold data |
| `src/build_release_files.py` | Builds the released baseline lexicon and tag counts from a private training split |
| `notebooks/01_data_preparation.ipynb` | Alignment, projection, lexicon verification and correction rounds |
| `notebooks/02_pos_tagger_bilstm_crf.ipynb` | Tagger training |
| `notebooks/04_evaluation.ipynb` | Reproduces every result in the paper from the released files |
| `notebooks/05_paper_tables.ipynb` | Generates the LaTeX results tables |
| `models/pos_tagger/` | Trained tagger (`best_model.pt`) and vocabularies |
| `data/lexicon/` | Mizo PoS lexicon, the 296 correction rules, and the baseline lexicon |
| `data/corpus/` | Corpus sample: the validation and test splits (801 sentences each) |
| `data/stats/` | Corpus statistics and training-split tag counts |
| `results/` | Evaluation outputs used in the paper |

## What is not included

The full parallel corpus and the silver-standard training split
(194,795 sentences) are not redistributed, because the underlying texts
remain under the copyright of their original publishers. Notebooks 01 and 02
therefore document the procedure but cannot be rerun end to end without your
own parallel data. Notebook 04 runs fully from the released files.

The external gold data of Ghosh et al. (2025) belongs to its authors. Obtain it
from them (see https://aclanthology.org/2025.law-1.18/) and place
`train.conllu`, `val.conllu` and `test.conllu` in `data/external/iiit/`.
Without it, notebook 04 skips the external evaluation.

## Reproducing the results

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # needed only for notebook 01
cd notebooks
jupyter notebook 04_evaluation.ipynb
```

## Tagging new text

```python
import sys; sys.path.insert(0, "src")
from model import load_tagger, tag_sentence
model, vocabs, device = load_tagger("models/pos_tagger")
tokens = "Aizawlah ka kal dawn".split()
print(list(zip(tokens, tag_sentence(model, vocabs, device, tokens))))
```

Input should be tokenized on whitespace,
with punctuation removed.

## Licenses

- Code (`src/`, `notebooks/`): MIT License, see `LICENSE`.
- Data and model (`data/`, `models/`): CC BY 4.0, see `LICENSE-DATA`.

## Citation

See `CITATION.cff`.
