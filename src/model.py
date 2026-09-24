"""
Mizo BiLSTM-CRF part-of-speech tagger.

Model classes extracted verbatim from notebooks/02_pos_tagger_bilstm_crf.ipynb
so that evaluation and inference do not require re-running training.

Typical use:

    from model import load_tagger, tag_sentence
    model, vocabs, device = load_tagger("../models/pos_tagger")
    print(tag_sentence(model, vocabs, device, ["Ka", "lawm", "e"]))
"""

import json
import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD = "<PAD>"
UNK = "<UNK>"
START_TAG = "<START>"
STOP_TAG = "<STOP>"
MAX_WORD_LEN = 30


class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, num_filters, kernel_size=3):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(char_emb_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, char_ids):
        batch_size, seq_len, max_word_len = char_ids.shape
        char_ids = char_ids.view(-1, max_word_len)
        char_emb = self.char_embedding(char_ids)
        char_emb = self.dropout(char_emb)
        char_emb = char_emb.transpose(1, 2)
        conv_out = self.conv(char_emb)
        pool_out = torch.max(conv_out, dim=2)[0]
        return pool_out.view(batch_size, seq_len, -1)

class CRF(nn.Module):
    def __init__(self, num_tags, start_tag_idx, stop_tag_idx, pad_tag_idx=0):
        super().__init__()
        self.num_tags = num_tags
        self.start_tag_idx = start_tag_idx
        self.stop_tag_idx = stop_tag_idx
        self.pad_tag_idx = pad_tag_idx
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.transitions.data[start_tag_idx, :] = -10000
        self.transitions.data[:, stop_tag_idx] = -10000
        self.transitions.data[pad_tag_idx, :] = -10000
        self.transitions.data[:, pad_tag_idx] = -10000
        self.transitions.data[pad_tag_idx, stop_tag_idx] = 0
        self.transitions.data[pad_tag_idx, pad_tag_idx] = 0
    
    def viterbi_decode(self, emissions, lengths):
        batch_size, seq_len, num_tags = emissions.shape
        viterbi = torch.full((batch_size, num_tags), -10000.0, device=emissions.device)
        viterbi[:, self.start_tag_idx] = 0.0
        backpointers = []
        for t in range(seq_len):
            emit_score = emissions[:, t, :]
            trans_score = self.transitions.unsqueeze(0)
            viterbi_exp = viterbi.unsqueeze(2)
            scores = viterbi_exp + trans_score
            best_scores, best_ids = scores.max(dim=1)
            viterbi_new = best_scores + emit_score
            mask = (t < lengths).float().unsqueeze(1)
            viterbi = viterbi_new * mask + viterbi * (1 - mask)
            backpointers.append(best_ids)
        viterbi += self.transitions[self.stop_tag_idx].unsqueeze(0)
        best_scores, best_last_tags = viterbi.max(dim=1)
        best_paths = []
        for b in range(batch_size):
            path = [best_last_tags[b].item()]
            for t in range(len(backpointers) - 1, 0, -1):
                if t < lengths[b]:
                    path.append(backpointers[t][b, path[-1]].item())
            path.reverse()
            best_paths.append(path[:lengths[b].item()])
        return best_paths

class BiLSTM_CRF(nn.Module):
    def __init__(self, word_vocab_size, char_vocab_size, tag_to_idx,
                 word_emb_dim=128, char_emb_dim=32, char_filters=64,
                 hidden_dim=256, num_layers=2, dropout=0.5):
        super().__init__()
        self.word_embedding = nn.Embedding(word_vocab_size, word_emb_dim, padding_idx=0)
        self.char_cnn = CharCNN(char_vocab_size, char_emb_dim, char_filters)
        self.input_dim = word_emb_dim + char_filters
        self.hidden_dim = hidden_dim
        self.num_tags = len(tag_to_idx)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            self.input_dim, hidden_dim // 2,
            num_layers=num_layers, bidirectional=True,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.hidden2tag = nn.Linear(hidden_dim, self.num_tags)
        self.crf = CRF(self.num_tags, tag_to_idx[START_TAG], tag_to_idx[STOP_TAG], tag_to_idx[PAD])
    
    def _get_emissions(self, word_ids, char_ids, lengths):
        word_emb = self.word_embedding(word_ids)
        char_emb = self.char_cnn(char_ids)
        combined = torch.cat([word_emb, char_emb], dim=2)
        combined = self.dropout(combined)
        packed = pack_padded_sequence(combined, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        return self.hidden2tag(lstm_out)
    
    def predict(self, word_ids, char_ids, lengths):
        emissions = self._get_emissions(word_ids, char_ids, lengths)
        return self.crf.viterbi_decode(emissions, lengths)

# ---------------------------------------------------------------- loading


def load_tagger(model_dir, device=None):
    """Load weights and vocabularies. Returns (model, vocabs, device)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(model_dir, "vocabularies.json"), encoding="utf-8") as f:
        v = json.load(f)
    vocabs = {
        "word_vocab": v["word_vocab"],
        "char_vocab": v["char_vocab"],
        "tag_to_idx": v["tag_to_idx"],
        "idx_to_tag": {int(k): t for k, t in v["idx_to_tag"].items()},
    }

    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"),
                      map_location=device)
    hp = ckpt["hyperparams"]

    model = BiLSTM_CRF(
        word_vocab_size=len(vocabs["word_vocab"]),
        char_vocab_size=len(vocabs["char_vocab"]),
        tag_to_idx=vocabs["tag_to_idx"],
        word_emb_dim=hp["word_emb_dim"],
        char_emb_dim=hp["char_emb_dim"],
        char_filters=hp["char_filters"],
        hidden_dim=hp["hidden_dim"],
        num_layers=hp["num_layers"],
        dropout=hp["dropout"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocabs, device


@torch.no_grad()
def tag_sentence(model, vocabs, device, tokens):
    """Tag one list of tokens. Returns a list of UD tags."""
    if not tokens:
        return []
    wv, cv, itt = vocabs["word_vocab"], vocabs["char_vocab"], vocabs["idx_to_tag"]

    word_ids = [wv.get(t.lower(), wv[UNK]) for t in tokens]
    char_ids = [[cv.get(c, cv[UNK]) for c in t[:MAX_WORD_LEN]] or [cv[UNK]]
                for t in tokens]
    width = max(len(c) for c in char_ids)
    char_ids = [c + [0] * (width - len(c)) for c in char_ids]

    w = torch.tensor([word_ids], dtype=torch.long, device=device)
    c = torch.tensor([char_ids], dtype=torch.long, device=device)
    lengths = torch.tensor([len(tokens)], dtype=torch.long, device=device)

    path = model.predict(w, c, lengths)[0]
    return [itt[i] for i in path]


def read_conll(path, tag_col=-1):
    """Read whitespace/tab-separated CoNLL. Returns [(tokens, tags), ...]."""
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
            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) < 2:
                continue
            toks.append(parts[0])
            tags.append(parts[tag_col])
    if toks:
        sents.append((toks, tags))
    return sents


def read_conllu(path):
    """Read 10-column CoNLL-U. UPOS is column 4. Skips multiword ranges."""
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
            if len(fields) < 4 or "-" in fields[0] or "." in fields[0]:
                continue
            toks.append(fields[1])
            tags.append(fields[3])
    if toks:
        sents.append((toks, tags))
    return sents


def score(pairs):
    """pairs = [(gold, pred), ...] -> dict of metrics."""
    n = len(pairs)
    if n == 0:
        return None
    acc = sum(1 for g, p in pairs if g == p) / n
    per = {}
    for t in sorted({g for g, _ in pairs}):
        tp = sum(1 for g, p in pairs if g == t and p == t)
        fp = sum(1 for g, p in pairs if g != t and p == t)
        fn = sum(1 for g, p in pairs if g == t and p != t)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        per[t] = {"precision": pr, "recall": rc, "f1": f1, "support": tp + fn}
    return {
        "accuracy": acc,
        "macro_f1": sum(v["f1"] for v in per.values()) / len(per),
        "weighted_f1": sum(v["f1"] * v["support"] for v in per.values()) / n,
        "tokens": n,
        "per_tag": per,
    }


def print_report(res, title):
    if res is None:
        print("\n%s: no tokens" % title)
        return
    print("\n===== %s =====" % title)
    print("Tokens:      %d" % res["tokens"])
    print("Accuracy:    %.4f (%.2f%%)" % (res["accuracy"], 100 * res["accuracy"]))
    print("Macro F1:    %.4f" % res["macro_f1"])
    print("Weighted F1: %.4f" % res["weighted_f1"])
    print("\n  %-8s %8s %8s %8s %8s"
          % ("Tag", "Prec", "Rec", "F1", "Support"))
    print("  " + "-" * 46)
    for t, v in sorted(res["per_tag"].items()):
        print("  %-8s %8.4f %8.4f %8.4f %8d"
              % (t, v["precision"], v["recall"], v["f1"], v["support"]))
