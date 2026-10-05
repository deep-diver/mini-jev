"""
A Kev-style decision head on Gemma 3 270M.

Reading Yes/No logits off a stock instruct model does not work at this size:
measured AUC 0.575 on synthetic data and 0.481 on real page text, both chance.
The open-source Jev reimplementations do not do that either -- kev, Laya and Von
all put a *trained* readout on a frozen backbone. This is that, at 270M.

Architecture, following kev's pointer head:

    State: <text>

    Question: <instructions>
    Options:
    first option <0>
    second option <1>
    Decision:

The sequence is run once. The hidden state at `Decision:` is the query, the
hidden state at each `<i>` marker is a key, and the option logits are their
scaled dot products. The marker trails its option because attention is causal:
placed in front it would summarise everything except the option it stands for. Nothing about the head is tied to a fixed label set, so
the options can be anything the caller passes at runtime -- which is the whole
requirement a page filter has.

The backbone stays frozen, so every hidden state is cached once and the head
then trains on tensors in seconds rather than on forward passes.

    python3 scripts/pointer_head.py cache      # one forward pass per example
    python3 scripts/pointer_head.py train      # fit the head + temperature
    python3 scripts/pointer_head.py eval       # held-out + the alphaXiv task
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "pointer")
MODEL = "google/gemma-3-270m-it"
MAX_OPTIONS = 8
LAYERS = (10, 13, 18)          # cache a few; the dev split picks one


# --------------------------------------------------------------------------
# Rendering: one typed request -> one string plus the marker positions we read

def render(state, instructions, options):
    lines = [f"State:\n{state}\n", f"Question: {instructions}", "Options:"]
    for i, o in enumerate(options):
        lines.append(f"{o} <{i}>")
    lines.append("Decision:")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Training data: public classification sets cast as typed requests, the same
# shape kev uses. Each record is (state, instructions, options, gold index).

def build_records(n_per=1800, seed=0):
    from datasets import load_dataset
    rng = random.Random(seed)
    recs = []

    def add(src, state, instr, options, gold):
        recs.append({"src": src, "state": state, "instructions": instr,
                     "options": options, "gold": gold})

    d = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=seed)
    names = ["World news", "Sports", "Business", "Science and technology"]
    for r in d.select(range(n_per)):
        add("ag_news", r["text"], "Which section does this story belong to?",
            names, r["label"])

    d = load_dataset("google/boolq", split="train").shuffle(seed=seed)
    for r in d.select(range(n_per)):
        add("boolq", f'{r["passage"]}\n\nQuestion: {r["question"]}',
            "Is the answer to the question yes?", ["Yes", "No"],
            0 if r["answer"] else 1)

    d = load_dataset("SetFit/sst5", split="train").shuffle(seed=seed)
    names = ["Very negative", "Negative", "Neutral", "Positive", "Very positive"]
    for r in d.select(range(min(n_per, len(d)))):
        add("sst5", r["text"], "How positive is this review?", names, r["label"])

    d = load_dataset("legacy-datasets/banking77", split="train").shuffle(seed=seed)
    feat = d.features["label"].names
    for r in d.select(range(n_per)):
        # 77 options would dominate the sequence; sample the gold plus distractors.
        others = [i for i in range(len(feat)) if i != r["label"]]
        pick = rng.sample(others, MAX_OPTIONS - 1) + [r["label"]]
        rng.shuffle(pick)
        add("banking77", r["text"], "What is this customer asking about?",
            [feat[i].replace("_", " ") for i in pick], pick.index(r["label"]))

    d = load_dataset("nyu-mll/glue", "mnli", split="train").shuffle(seed=seed)
    names = ["It follows", "It is unrelated", "It contradicts"]
    for r in d.select(range(n_per)):
        add("mnli", f'Statement: {r["premise"]}\nClaim: {r["hypothesis"]}',
            "What is the relationship of the claim to the statement?",
            names, r["label"])

    rng.shuffle(recs)
    return recs


# --------------------------------------------------------------------------

def load_backbone():
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float32).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model, device


@torch.no_grad()
def encode(tok, model, device, records, layers=LAYERS, batch=8, log=None):
    """Return {layer: (N, 1 + MAX_OPTIONS, hidden)} -- query first, then keys."""
    hid = model.config.hidden_size
    outs = {L: np.zeros((len(records), 1 + MAX_OPTIONS, hid), dtype=np.float32)
            for L in layers}
    mask = np.zeros((len(records), MAX_OPTIONS), dtype=bool)

    for start in range(0, len(records), batch):
        chunk = records[start:start + batch]
        texts, marks = [], []
        for r in chunk:
            texts.append(render(r["state"], r["instructions"], r["options"]))
            marks.append(len(r["options"]))
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=768, add_special_tokens=True,
                  return_offsets_mapping=True).to(device)
        offsets = enc.pop("offset_mapping")
        out = model(**enc, output_hidden_states=True)

        for bi, r in enumerate(chunk):
            n_tok = int(enc["attention_mask"][bi].sum())
            # Locate markers by character offset rather than by token id: the
            # tokenizer merges the preceding space into "<", so searching for
            # the id sequence of "<0>" silently finds nothing.
            off = offsets[bi, :n_tok].tolist()
            pos = []
            for i in range(len(r["options"])):
                ch = texts[bi].find(f"<{i}>")
                pos.append(token_at(off, ch + 2) if ch >= 0 else n_tok - 1)
            for L in layers:
                h = out.hidden_states[L][bi]
                outs[L][start + bi, 0] = h[n_tok - 1].float().cpu().numpy()
                for i, p in enumerate(pos):
                    outs[L][start + bi, 1 + i] = h[p].float().cpu().numpy()
            mask[start + bi, :len(r["options"])] = True

        if log and (start // batch) % 50 == 0:
            print(f"   {start + len(chunk):6d}/{len(records)}", flush=True)
    return outs, mask


def token_at(offsets, char):
    """Index of the token whose span covers `char` (last token if past the end)."""
    for i, (a, b) in enumerate(offsets):
        if a <= char < b:
            return i
    return len(offsets) - 1


# --------------------------------------------------------------------------

class PointerHead(nn.Module):
    """logit_i = <W_q h_decision, W_k h_option_i> / sqrt(d)."""

    def __init__(self, hidden, dim=256):
        super().__init__()
        self.q = nn.Linear(hidden, dim, bias=False)
        self.k = nn.Linear(hidden, dim, bias=False)
        self.norm = nn.LayerNorm(hidden)
        self.dim = dim

    def forward(self, states, mask):
        x = self.norm(states)
        q = self.q(x[:, 0])                      # (B, dim)
        k = self.k(x[:, 1:])                     # (B, MAX_OPTIONS, dim)
        logits = (k @ q.unsqueeze(-1)).squeeze(-1) / self.dim ** 0.5
        return logits.masked_fill(~mask, -1e4)


def fit_head(Xtr, mtr, ytr, Xdv, mdv, ydv, dim, epochs, lr=3e-3, bs=256):
    head = PointerHead(Xtr.shape[-1], dim)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    best, best_state = -1.0, None
    for ep in range(epochs):
        head.train()
        idx = torch.randperm(len(Xtr))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            loss = nn.functional.cross_entropy(head(Xtr[j], mtr[j]), ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            a = (head(Xdv, mdv).argmax(-1) == ydv).float().mean().item()
        if a > best:
            best, best_state = a, {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    return head, best


def fit_temperature(logits, gold):
    """One scalar, fitted by NLL on held-out data. Every Jev reimplementation
    ships one; raw head confidence is systematically too high."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = nn.functional.cross_entropy(logits / logT.exp(), gold)
        loss.backward()
        return loss
    opt.step(closure)
    return float(logT.exp())


def ece(p, gold, bins=15):
    conf, pred = p.max(-1)
    correct = (pred == gold).float()
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean() * (conf[m].mean() - correct[m].mean()).abs()
    return float(e)


def brier(p, gold, mask):
    t = torch.zeros_like(p)
    t[torch.arange(len(gold)), gold] = 1.0
    return float((((p - t) ** 2) * mask).sum(-1).mean())


def eval_alphaxiv(args):
    """Out of domain by construction: no page text, no relevance question and
    no paper title appeared anywhere in training."""
    import itertools
    ck = torch.load(os.path.join(OUT, "head.pt"), weights_only=False)
    head = PointerHead(640, ck["dim"])
    head.load_state_dict(ck["state"])
    head.eval()
    L, T = ck["layer"], ck["temperature"]

    data = json.load(open(os.path.join(ROOT, "data", "eval", "alphaxiv_multi.json")))
    titles = data["titles"]
    tok, model, device = load_backbone()

    def auc(scores, y):
        pos = [a for a, t in zip(scores, y) if t]
        neg = [a for a, t in zip(scores, y) if not t]
        return sum((a > b) + 0.5 * (a == b)
                   for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))

    def run(criterion, flip):
        opts = ["Something else", criterion] if flip else [criterion, "Something else"]
        want = 1 if flip else 0
        recs = [{"state": t, "instructions": "Which describes this title?",
                 "options": opts, "gold": 0} for t in titles]
        st, mask = encode(tok, model, device, recs, layers=(L,))
        with torch.no_grad():
            p = torch.softmax(head(torch.tensor(st[L]), torch.tensor(mask)) / T, -1)
        return p[:, want].tolist()

    print(f"{'crit':4s} {'as given':>10s} {'reversed':>10s} {'averaged':>10s}")
    print("-" * 38)
    rows = []
    for k, c in data["criteria"].items():
        y = np.zeros(len(titles), dtype=bool)
        y[c["positive"]] = True
        a = run(c["text"], False)
        b = run(c["text"], True)
        avg = [(x + z) / 2 for x, z in zip(a, b)]
        rows.append((auc(a, y), auc(b, y), auc(avg, y)))
        print(f"{k:4s} {rows[-1][0]:8.3f} {rows[-1][1]:8.3f} {rows[-1][2]:12.3f}")
    m = np.array(rows).mean(0)
    print("-" * 38)
    print(f"{'mean':4s} {m[0]:10.3f} {m[1]:10.3f} {m[2]:10.3f}")
    print("\nfor comparison, same 179 titles:")
    print("   zero-shot Yes/No logits     0.564 / 0.588 / 0.304   mean 0.485")
    print("   the shipped linear probe    0.481 on criterion A")
    print("   open-vocabulary prompting   0.753 / 0.799 / 0.854   mean 0.802 (not Jev)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["cache", "train", "eval"])
    ap.add_argument("--n-per", type=int, default=1800)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--dim", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if args.stage == "cache":
        recs = build_records(args.n_per)
        print(f"{len(recs)} typed requests from "
              f"{len(set(r['src'] for r in recs))} datasets")
        tok, model, device = load_backbone()
        print(f"backbone frozen on {device}; one forward pass each")
        states, mask = encode(tok, model, device, recs, log=True)
        np.savez_compressed(
            os.path.join(OUT, "train_states.npz"),
            mask=mask, gold=np.array([r["gold"] for r in recs]),
            src=np.array([r["src"] for r in recs]),
            **{f"L{L}": v for L, v in states.items()})
        print(f"-> {OUT}/train_states.npz")
        return 0

    d = np.load(os.path.join(OUT, "train_states.npz"), allow_pickle=True)
    gold = torch.tensor(d["gold"], dtype=torch.long)
    mask = torch.tensor(d["mask"])
    src = d["src"]
    n = len(gold)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_te = n // 10
    te, dev, tr = perm[:n_te], perm[n_te:2 * n_te], perm[2 * n_te:]

    if args.stage == "train":
        best = None
        for L in LAYERS:
            X = torch.tensor(d[f"L{L}"])
            head, dev_acc = fit_head(X[tr], mask[tr], gold[tr],
                                     X[dev], mask[dev], gold[dev],
                                     args.dim, args.epochs)
            print(f"  layer {L:2d}   dev accuracy {dev_acc:.3f}")
            if best is None or dev_acc > best[1]:
                best = (L, dev_acc, head)
        L, dev_acc, head = best

        X = torch.tensor(d[f"L{L}"])
        with torch.no_grad():
            dev_logits = head(X[dev], mask[dev])
            T = fit_temperature(dev_logits, gold[dev])
            te_logits = head(X[te], mask[te]) / T
            p = torch.softmax(te_logits, -1)
        acc = (p.argmax(-1) == gold[te]).float().mean().item()
        print(f"\nchosen: layer {L}, temperature {T:.2f}")
        print(f"test accuracy {acc:.3f} · ECE {ece(p, gold[te]):.3f} "
              f"· Brier {brier(p, gold[te], mask[te]):.3f}")
        print(f"{'source':12s} {'n':>5s} {'acc':>7s}")
        for s_ in sorted(set(src)):
            m = torch.tensor([src[i] == s_ for i in te.tolist()])
            if m.any():
                print(f"{s_:12s} {int(m.sum()):5d} "
                      f"{(p.argmax(-1)[m] == gold[te][m]).float().mean():7.3f}")
        torch.save({"state": head.state_dict(), "layer": L, "dim": args.dim,
                    "temperature": T}, os.path.join(OUT, "head.pt"))
        print(f"\n{sum(q.numel() for q in head.parameters()):,} parameters "
              f"-> {OUT}/head.pt")
        return 0

    return eval_alphaxiv(args)


if __name__ == "__main__":
    sys.exit(main())
