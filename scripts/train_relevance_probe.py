"""
Fit and save the relevance probe for Gemma 3 270M.

The model encodes "does this element match the user's criterion" in its
residual stream but cannot express it through the Yes/No token logits, which
come back at ~0.999 for everything. This fits a single linear readout on the
layer-15 residual stream instead: 641 parameters, and the forward pass gets
*cheaper* because the last three layers and the 262k-way vocabulary projection
are never computed.

Scored leave-one-criterion-out, because the criterion is whatever the user
types at call time.

    python3 scripts/train_relevance_probe.py
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from probe_sweep import CACHE  # noqa: E402

OUT = os.path.join(ROOT, "data", "surfmate", "relevance_probe.npz")
LAYER = 15
RIDGE = 100.0


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b)
               for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))


def centre_by_group(X, groups):
    """Subtract each group's own mean.

    At inference the group is the page: a filter always scores every candidate
    element at once, so the page mean is free. Removing it strips the part of
    the residual stream that is about answering a Yes/No question at all, which
    is identical for every element and swamps the part that is about this
    element.
    """
    out = X.astype(np.float64).copy()
    for g in set(groups):
        m = groups == g
        out[m] -= out[m].mean(0)
    return out


def fit(X, y, ridge):
    A = np.c_[X, np.ones(len(X))]
    R = ridge * np.eye(A.shape[1])
    R[-1, -1] = 0.0
    return np.linalg.solve(A.T @ A + R, A.T @ (y * 2.0 - 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--ridge", type=float, default=RIDGE)
    args = ap.parse_args()

    if not os.path.exists(CACHE):
        print(f"missing {CACHE}; run: python3 scripts/probe_sweep.py --extract")
        return 1

    d = np.load(CACHE, allow_pickle=True)
    H = d["states"][:, args.layer].astype(np.float64)
    y = d["labels"]
    crits = d["crits"]
    ps = d["ps"]

    # Fixed centre, not the batch mean. Centring on whatever is being scored
    # right now works only while the batch is a mixture: on a feed where most
    # items are on-topic, the topic becomes the mean and the relevant items end
    # up *below* zero. Measured on a homogeneous batch, batch centring put the
    # positives at -0.50 while a fixed centre put them at +0.56, for the same
    # leave-one-criterion-out AUC (0.977 vs 0.971).
    mu = H.mean(0)
    Xc = H - mu
    scale = Xc.std(0) + 1e-6
    X = Xc / scale

    # Honest score first: every criterion held out in turn.
    scores = np.zeros(len(y))
    for c in sorted(set(crits)):
        te = crits == c
        w_tr = fit(X[~te], y[~te], args.ridge)
        scores[te] = np.c_[X[te], np.ones(te.sum())] @ w_tr
    per = {c: auc(scores[crits == c], y[crits == c]) for c in sorted(set(crits))}

    print(f"layer {args.layer}, ridge {args.ridge}, {len(y)} pairs, "
          f"{len(set(crits))} criteria")
    print(f"  Yes/No logit read : AUC {auc(ps, y):.3f}")
    print(f"  probe, unseen crit: AUC {np.mean(list(per.values())):.3f}  "
          f"(worst {min(per, key=per.get)} {min(per.values()):.3f})")

    # Ship a probe fitted on everything; the held-out numbers above are what it
    # is expected to do on a criterion it has not seen.
    w = fit(X, y, args.ridge)
    np.savez(OUT, w=w, scale=scale, mean=mu, layer=args.layer, ridge=args.ridge,
             loco_auc=np.mean(list(per.values())))
    print(f"\n{len(w)} parameters -> {OUT}")
    print("per-criterion AUC: " + json.dumps({k: round(v, 3) for k, v in per.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
