"""
Build the derived files that let the released notebooks run without the
full training corpus, which is not redistributed.

Run once, from the repository root, pointing at your private copy of
train.conll:

    python src/build_release_files.py C:\\Users\\Haulai\\mizo-pos-private\\data\\corpus\\train.conll

Writes:
    data/lexicon/baseline_lexicon.tsv   word <TAB> most frequent tag
    data/stats/train_tag_counts.json    tag counts over the training split

Both contain word types and tags only, no sentences.
"""
import sys, json, collections
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from model import read_conll

if len(sys.argv) != 2:
    sys.exit("usage: python src/build_release_files.py PATH/TO/train.conll")

src = Path(sys.argv[1])
train = read_conll(src)
counts = collections.defaultdict(collections.Counter)
for toks, tags in train:
    for t, g in zip(toks, tags):
        counts[t.lower()][g] += 1

# Tag counts are taken line by line, as in the paper's Table 3, so that every
# token line in the file is counted.
totals = collections.Counter()
with open(src, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            totals[line.rstrip("\n").rsplit("\t", 1)[-1].strip()] += 1

lex_path = ROOT / "data" / "lexicon" / "baseline_lexicon.tsv"
with open(lex_path, "w", encoding="utf-8", newline="\n") as f:
    for w in sorted(counts):
        f.write("%s\t%s\n" % (w, counts[w].most_common(1)[0][0]))

stats_path = ROOT / "data" / "stats" / "train_tag_counts.json"
stats_path.write_text(json.dumps({
    "sentences": len(train),
    "tokens": sum(totals.values()),
    "default_tag": totals.most_common(1)[0][0],
    "tag_counts": dict(totals.most_common()),
}, indent=2), encoding="utf-8")

print("sentences: %d   tokens: %d" % (len(train), sum(totals.values())))
print("lexicon entries: %d  -> %s" % (len(counts), lex_path))
print("tag counts           -> %s" % stats_path)
