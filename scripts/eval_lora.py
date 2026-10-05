"""
Score the LoRA + pointer head on the held-out alphaXiv task.

Out of domain by construction: the training mix is news, encyclopedia entries,
forum questions, bank messages and reviews. No paper title, no page text and no
relevance question appears in it. That is the only way to find out whether the
decision interface generalises, as opposed to whether it memorised a corpus --
the mistake that produced a probe scoring 0.977 in training and 0.481 in the
field.

    python3 scripts/eval_lora.py --ckpt lora_head.npz --data alphaxiv_multi.json
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from train_lora_tpu import (  # noqa: E402
    build_batchable, load_weights, make_model, stack_layers)


def auc(scores, y):
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    return sum((a > b) + 0.5 * (a == b)
               for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="lora_head.npz")
    ap.add_argument("--data", default="alphaxiv_multi.json")
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from transformers import AutoTokenizer

    ck = np.load(args.ckpt)
    rank, dim = int(ck["rank"]), int(ck["dim"])
    scale = float(ck["alpha"]) / rank
    params = {
        "lora": {k[5:]: jnp.asarray(ck[k]) for k in ck.files if k.startswith("lora.")},
        "head": {k[5:]: jnp.asarray(ck[k]) for k in ck.files if k.startswith("head.")},
    }
    print(f"checkpoint: rank {rank}, dim {dim}, dev acc {float(ck['dev_acc']):.3f}")

    tok = AutoTokenizer.from_pretrained(args.model)
    weights = load_weights(args.model, jnp)
    const = {"stacked": stack_layers(weights, jnp),
             "embed": weights["model.embed_tokens.weight"],
             "final_norm": weights["model.norm.weight"]}
    logits_fn, _ = make_model(jnp, jax)

    zero = {k: jnp.zeros_like(v) for k, v in params["lora"].items()}

    def score(records, lora):
        p = {"lora": lora, "head": params["head"]}
        batch, kept = build_batchable(tok, records, args.max_len, 8)
        out = []
        n = len(batch["gold"])
        for i in range(0, n, args.batch):
            j = np.arange(i, min(i + args.batch, n))
            b = {k: jnp.asarray(v[j]) for k, v in batch.items()}
            lg = logits_fn(p, const, b, args.max_len, scale)
            out += jax.nn.softmax(lg, -1)[:, 0].tolist()   # P(first option)
        return out, kept

    data = json.load(open(args.data))
    titles = data["titles"]
    rows = {}
    for key, c in data["criteria"].items():
        y = np.zeros(len(titles), dtype=bool)
        y[c["positive"]] = True
        recs = [{"state": t, "instructions": "Which describes this title?",
                 "options": [c["text"], "Something else"], "gold": 0}
                for t in titles]
        for name, lora in (("adapter off (head only)", zero), ("adapter on", params["lora"])):
            s, kept = score(recs, lora)
            assert len(kept) == len(titles), f"{len(kept)} of {len(titles)} kept"
            rows.setdefault(name, {})[key] = auc(s, y)

    print(f"\n{'setting':24s} " + "".join(f"{k:>8s}" for k in data["criteria"])
          + f"{'mean':>8s}")
    print("-" * 56)
    for name, per in rows.items():
        v = [per[k] for k in data["criteria"]]
        print(f"{name:22s} " + "".join(f"{x:8.3f}" for x in v) + f"{np.mean(v):8.3f}")
    print("-" * 56)
    print("for comparison, same 179 titles and labels:")
    print("   zero-shot Yes/No logits                 mean 0.485")
    print("   frozen backbone + pointer head (9k)     mean 0.704")
    return 0


if __name__ == "__main__":
    sys.exit(main())
