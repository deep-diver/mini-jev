"""
Can the label words be derived from the criterion automatically?

Hand-picking `RL` / `other` got AUC 0.829 on one criterion, but a filter takes
whatever the user types. So the label words have to come out of the criterion
with no human in the loop -- and the rule has to be fixed before it is scored,
on more than one criterion, or this just repeats the mistake that produced a
0.977 probe which measured 0.481 in the field.

Three criteria are labelled over the same 179 alphaXiv titles. The rule never
sees the labels.

    python3 scripts/auto_labels.py
"""

import argparse
import itertools
import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "eval", "alphaxiv_multi.json")
MODEL = "google/gemma-3-270m-it"

# Words that carry no topic. Deliberately short: a long hand-tuned list would
# be the same overfitting through a side door.
STOP = {"this", "is", "about", "or", "and", "the", "a", "an", "with", "of",
        "for", "in", "on", "to", "that", "it", "page", "element", "user",
        "looking", "something", "else", "pure", "general", "new"}

# The format-teaching shots use a topic unrelated to any criterion under test,
# so they show the model the shape of the answer without leaking the answer.
FORMAT_SHOTS = [
    ("Sourdough Fermentation Times Across Flour Types", "cooking"),
    ("Low-Rank Attention Kernels for Long-Context Inference", "other"),
    ("A Field Guide to Alpine Wildflowers", "plants"),
    ("Spectral Gaps of Random Regular Graphs", "other"),
]


def criterion_words(text):
    """Content words of the criterion, in order, deduplicated."""
    out = []
    for w in re.findall(r"[A-Za-z]+", text):
        lw = w.lower()
        if lw in STOP or len(lw) < 3:
            continue
        if lw not in [o.lower() for o in out]:
            out.append(w)
    return out


def auc(scores, y):
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    return sum((a > b) + 0.5 * (a == b)
               for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    d = json.load(open(DATA))
    titles = d["titles"]

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32).to(device).eval()

    def first_ids(words):
        seen, out = set(), []
        for w in words:
            i = tok.encode(w, add_special_tokens=False)[0]
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out

    OTHER = first_ids(["other"])

    @torch.no_grad()
    def score(prompts, pos, neg):
        out = []
        for i in range(0, len(prompts), 16):
            enc = tok(prompts[i:i + 16], return_tensors="pt", padding=True,
                      add_special_tokens=True).to(device)
            last = enc["attention_mask"].sum(1) - 1
            rows = torch.arange(len(last), device=device)
            lg = model(**enc).logits[rows, last].float()
            a = lg[:, pos].max(1).values
            b = lg[:, neg].max(1).values
            out += torch.softmax(torch.stack([a, b], 1), 1)[:, 0].tolist()
        return out

    def p_zero(t, _words):
        return f'Title: "{t}"\nLabel:'

    def p_format(t, _words):
        block = "".join(f'Title: "{s}"\nLabel: {l}\n\n' for s, l in FORMAT_SHOTS)
        return block + f'Title: "{t}"\nLabel:'

    def p_stated(t, words):
        block = "".join(f'Title: "{s}"\nLabel: {l}\n\n' for s, l in FORMAT_SHOTS)
        return (f"Label each paper title with {words[0].lower()} or other.\n\n"
                + block + f'Title: "{t}"\nLabel:')

    BUILDERS = {"0-shot pattern": p_zero,
                "format shots (unrelated topic)": p_format,
                "format shots + one instruction": p_stated}

    print(f"{len(titles)} titles · {args.model} · {device}\n")
    table = {}
    for key, c in d["criteria"].items():
        words = criterion_words(c["text"])
        pos = first_ids(words)
        y = np.zeros(len(titles), dtype=bool)
        y[c["positive"]] = True
        print(f'[{key}] "{c["text"]}"')
        print(f'     label words from the criterion: {words}  '
              f'({len(pos)} positive tokens vs "other")')
        for name, build in BUILDERS.items():
            s = score([build(t, words) for t in titles], pos, OTHER)
            a = auc(s, y)
            table.setdefault(name, {})[key] = a
            print(f"     {name:22s} AUC {a:.3f}")
        print()

    print(f"{'prompt':30s} " + "".join(f"{k:>8s}" for k in d["criteria"]) + f"{'mean':>8s}")
    print("-" * 58)
    for name, per in table.items():
        vals = [per[k] for k in d["criteria"]]
        print(f"{name:24s} " + "".join(f"{v:8.3f}" for v in vals)
              + f"{np.mean(vals):8.3f}")
    print("-" * 58)
    print("for reference: hand-picked RL/other scores 0.829 on A; a probe on the "
          "hidden states scores 0.839")
    return 0


if __name__ == "__main__":
    sys.exit(main())
