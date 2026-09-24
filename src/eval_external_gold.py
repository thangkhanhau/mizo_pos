# ============================================================
# External gold-standard evaluation: Ghosh et al. (2025) Mizo UD PoS
#
# Paste this as a NEW CELL at the END of
#   "2 - Mizo PoS Tagger using BiLSTM-CRF.ipynb"
# and run it. It reuses model, word_vocab, char_vocab, tag_to_idx,
# idx_to_tag and device that the notebook has already defined and loaded.
#
# Put the downloaded .conllu files in a folder named  iiit_pos_data
# next to the notebook. All three splits are used: the model was
# trained on none of them, so there is no leakage in evaluating on
# the full 502 sentences rather than only their 50-sentence test split.
# ============================================================

import os
import json
import collections
import torch

GOLD_DIR = "iiit_pos_data"
GOLD_FILES = ["train.conllu", "val.conllu", "test.conllu"]
MAX_WORD_LEN = 30
OUT = os.path.join(MODEL_DIR, "external_gold_results.json")


# ---------------------------------------------------------- read CoNLL-U
def read_conllu(path):
    """Yield (tokens, tags) per sentence. Skips comments and multiword ranges."""
    sents, toks, tags = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                if toks:
                    sents.append((toks, tags))
                    toks, tags = [], []
                continue
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            if "-" in fields[0] or "." in fields[0]:
                continue          # multiword token / empty node
            toks.append(fields[1])
            tags.append(fields[3])
    if toks:
        sents.append((toks, tags))
    return sents


gold = []
for name in GOLD_FILES:
    path = os.path.join(GOLD_DIR, name)
    if os.path.exists(path):
        s = read_conllu(path)
        gold += s
        print("  %-14s %4d sentences" % (name, len(s)))
    else:
        print("  %-14s MISSING - skipped" % name)

print("\nTotal gold sentences: %d" % len(gold))
print("Total gold tokens:    %d" % sum(len(t) for t, _ in gold))


# ---------------------------------------------------------- tag the data
@torch.no_grad()
def tag_sentence(tokens):
    word_ids = [word_vocab.get(t.lower(), word_vocab[UNK]) for t in tokens]
    char_ids = [[char_vocab.get(c, char_vocab[UNK]) for c in t[:MAX_WORD_LEN]]
                for t in tokens]
    maxlen = max(len(c) for c in char_ids)
    char_pad = [c + [0] * (maxlen - len(c)) for c in char_ids]

    w = torch.tensor([word_ids], dtype=torch.long, device=device)
    c = torch.tensor([char_pad], dtype=torch.long, device=device)
    lens = torch.tensor([len(tokens)], dtype=torch.long)

    paths = model.predict(w, c, lens)
    return [idx_to_tag[str(i)] if isinstance(next(iter(idx_to_tag)), str)
            else idx_to_tag[i] for i in paths[0]]


model.eval()
pairs = []                       # (gold_tag, pred_tag)
for tokens, tags in gold:
    if not tokens:
        continue
    pred = tag_sentence(tokens)
    pairs += list(zip(tags, pred[:len(tags)]))

print("Tagged %d tokens" % len(pairs))


# ---------------------------------------------------------- score
def score(pairs, label):
    if not pairs:
        print("\n%s: no tokens" % label)
        return None
    n = len(pairs)
    acc = sum(1 for g, p in pairs if g == p) / n
    tags = sorted({g for g, _ in pairs})
    per = {}
    for t in tags:
        tp = sum(1 for g, p in pairs if g == t and p == t)
        fp = sum(1 for g, p in pairs if g != t and p == t)
        fn = sum(1 for g, p in pairs if g == t and p != t)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[t] = {"precision": pr, "recall": rc, "f1": f1, "support": tp + fn}
    macro = sum(v["f1"] for v in per.values()) / len(per)
    weighted = sum(v["f1"] * v["support"] for v in per.values()) / n

    print("\n===== %s =====" % label)
    print("Tokens:      %d" % n)
    print("Accuracy:    %.4f (%.2f%%)" % (acc, 100 * acc))
    print("Macro F1:    %.4f" % macro)
    print("Weighted F1: %.4f" % weighted)
    print("\n  %-8s %8s %8s %8s %8s" % ("Tag", "Prec", "Rec", "F1", "Support"))
    print("  " + "-" * 46)
    for t in tags:
        v = per[t]
        print("  %-8s %8.4f %8.4f %8.4f %8d"
              % (t, v["precision"], v["recall"], v["f1"], v["support"]))
    return {"accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
            "tokens": n, "per_tag": per}


full = score(pairs, "ALL TOKENS")
nopunct = score([(g, p) for g, p in pairs if g != "PUNCT"],
                "PUNCTUATION EXCLUDED")


# ------------------------------------------- where the conventions differ
print("\n===== TOP CONFUSIONS (gold -> predicted) =====")
conf = collections.Counter((g, p) for g, p in pairs if g != p and g != "PUNCT")
for (g, p), k in conf.most_common(15):
    print("  %-6s -> %-6s  %5d" % (g, p, k))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"all_tokens": full, "punct_excluded": nopunct,
               "confusions": {"%s->%s" % k: v for k, v in conf.most_common(30)}},
              f, indent=2)
print("\nSaved: %s" % OUT)
