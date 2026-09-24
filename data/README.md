# Data

- `corpus/dev.conll`, `corpus/test.conll`: validation and test splits, 801
  sentences each, drawn from the manually reviewed core. One token per line,
  `token<TAB>UD tag`, blank line between sentences.
- `lexicon/mizo_lexicon.tsv`, `mizo_words.tsv`, `mizo_phrases.tsv`: Mizo PoS
  lexicon compiled by the first author from printed and online dictionaries
  and word lists.
- `lexicon/correction_rules.json`: the 296 native-speaker correction rules.
- `lexicon/baseline_lexicon.tsv`: most frequent training tag per word type,
  used by the lexicon baseline.
- `stats/`: corpus statistics and training-split tag counts.
- `external/iiit/`: place the gold data of Ghosh et al. (2025) here
  (not redistributed; see the main README).

The full training split is not redistributed. See the main README.

License: CC BY 4.0 (see `LICENSE-DATA`).
