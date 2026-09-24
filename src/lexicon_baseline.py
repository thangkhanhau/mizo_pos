"""
Lexicon-based PoS tagger baseline for Mizo.

Builds a most-frequent-tag lexicon from the TRAINING split only, then
evaluates it on the held-out test split. Building from the training split
alone is what keeps the comparison against the BiLSTM-CRF honest -- a
lexicon built over the whole corpus would have already seen the test tokens.

Run from:  Experiment 1 - Data Preparation & PoS Tagging/
    python lexicon_baseline.py

Writes data2/model/lexicon_baseline_results.json in the same shape as
test_results.json, so the two are directly comparable.
"""

import json
import os
from collections import Counter, defaultdict

DATA_DIR = "data2"
TRAIN = os.path.join(DATA_DIR, "train_combined.conll")
TEST = os.path.join(DATA_DIR, "test.conll")
OUT = os.path.join(DATA_DIR, "model", "lexicon_baseline_results.json")


def read_conll(path):
    """Yield (token, tag) pairs. Blank lines separate sentences."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                parts = line.split()
            if len(parts) < 2:
                continue
            yield parts[0], parts[-1]


# ---------------------------------------------------------------- build
counts = defaultdict(Counter)
tag_totals = Counter()
for tok, tag in read_conll(TRAIN):
    counts[tok.lower()][tag] += 1
    tag_totals[tag] += 1

lexicon = {w: c.most_common(1)[0][0] for w, c in counts.items()}
default_tag = tag_totals.most_common(1)[0][0]

print("Lexicon entries:      %d" % len(lexicon))
print("OOV default tag:      %s" % default_tag)

# ----------------------------------------------------------- evaluate
gold = []
pred = []
oov = 0
for tok, tag in read_conll(TEST):
    p = lexicon.get(tok.lower())
    if p is None:
        p = default_tag
        oov += 1
    gold.append(tag)
    pred.append(p)

n = len(gold)
correct = sum(1 for g, p in zip(gold, pred) if g == p)
accuracy = correct / n

tags = sorted(set(gold))
per_tag = {}
for t in tags:
    tp = sum(1 for g, p in zip(gold, pred) if g == t and p == t)
    fp = sum(1 for g, p in zip(gold, pred) if g != t and p == t)
    fn = sum(1 for g, p in zip(gold, pred) if g == t and p != t)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    per_tag[t] = {"precision": prec, "recall": rec, "f1": f1,
                  "support": tp + fn}

macro_f1 = sum(v["f1"] for v in per_tag.values()) / len(per_tag)
weighted_f1 = sum(v["f1"] * v["support"] for v in per_tag.values()) / n

print("\nTest tokens:          %d" % n)
print("OOV tokens:           %d (%.1f%%)" % (oov, 100 * oov / n))
print("Accuracy:             %.4f (%.2f%%)" % (accuracy, 100 * accuracy))
print("Macro F1:             %.4f" % macro_f1)
print("Weighted F1:          %.4f" % weighted_f1)

print("\n  %-8s %9s %9s %9s %9s" % ("Tag", "Prec", "Rec", "F1", "Support"))
print("  " + "-" * 48)
for t in tags:
    v = per_tag[t]
    print("  %-8s %9.4f %9.4f %9.4f %9d"
          % (t, v["precision"], v["recall"], v["f1"], v["support"]))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"accuracy": accuracy,
               "macro_f1": macro_f1,
               "weighted_f1": weighted_f1,
               "lexicon_size": len(lexicon),
               "oov_tokens": oov,
               "test_tokens": n,
               "default_tag": default_tag,
               "per_tag_f1": per_tag}, f, indent=2)
print("\nSaved: %s" % OUT)
