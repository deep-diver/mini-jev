"""
Can one linear head serve criteria it has never seen?

Reading the Yes/No token logits out of Gemma 3 270M gives nothing: every answer
comes back at ~0.999 regardless of the input. But a linear probe on the final
hidden state of the *same* forward pass separates the same examples perfectly,
so the model does encode the answer -- the language-model head just cannot
express it.

That only helps if one head works for arbitrary criteria, because the criterion
is supplied by the user at call time. The criterion is part of the prompt, so
the hidden state should encode "does this element match *that* criterion"
rather than any one fixed topic. This checks exactly that by splitting on
CRITERION: the probe is fitted on some criteria and scored on criteria it has
never been trained on.

    python3 scripts/probe_generalisation.py
"""

import argparse
import itertools
import os
import sys

import numpy as np
import torch

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

BASE = "google/gemma-3-270m-it"

# A pool of page-element texts with a topic tag. Relevance is then a function of
# (item, criterion), so the same item is positive for one criterion and negative
# for the next -- which is what stops the probe from simply memorising items.
ITEMS = [
    ("ai", "Nvidia unveils new Blackwell GPU architecture for AI training"),
    ("ai", "How large language models are changing software engineering"),
    ("ai", "OpenAI releases a smaller, faster reasoning model"),
    ("ai", "Quantization brings a 70B model onto a laptop"),
    ("ai", "A survey of retrieval augmented generation techniques"),
    ("ai", "Benchmarking int8 inference on Apple silicon"),
    ("food", "Why chef Ferran Adria is the subject of a new documentary"),
    ("food", "A history of sourdough baking in northern Europe"),
    ("food", "The best ramen shops in the city, ranked"),
    ("food", "How to braise short ribs without drying them out"),
    ("food", "Restaurant review: a tasting menu worth the queue"),
    ("food", "Fermentation basics: kimchi, miso and vinegar"),
    ("sport", "Manchester City beat Arsenal in a late comeback"),
    ("sport", "The marathon world record falls again in Berlin"),
    ("sport", "How a rookie point guard changed the season"),
    ("sport", "Olympic swimming trials: who qualified overnight"),
    ("sport", "Tennis: a five-set final that lasted four hours"),
    ("sport", "Cycling team announces its Tour de France roster"),
    ("finance", "FCC lets Paramount sell an equity stake to a foreign investor"),
    ("finance", "Treasury yields climb as inflation data surprises"),
    ("finance", "A startup raises $40M in a round led by Sequoia"),
    ("finance", "Why the central bank held rates steady this month"),
    ("finance", "Quarterly earnings beat expectations on cloud revenue"),
    ("finance", "Retail investors pile into a newly listed company"),
    ("nature", "Weeping whales: stillborn humpback whale grieving documented"),
    ("nature", "T. rex teeth indicate it ran as warm as an elephant"),
    ("nature", "Rings around a tiny body have changed over the past decade"),
    ("nature", "Bear-y good neighbours: humans and animals share a town"),
    ("nature", "A coral reef recovers faster than anyone predicted"),
    ("nature", "Migrating birds navigate using a magnetic sense"),
    ("health", "Finding the cells that put our brain to sleep"),
    ("health", "Learning another language may keep your mind sharp"),
    ("health", "A new trial for a once-weekly diabetes drug"),
    ("health", "Why sleep debt cannot simply be repaid at the weekend"),
    ("health", "Physiotherapy beats surgery for some knee injuries"),
    ("health", "Vitamin D supplements show no effect in a large study"),
    ("boiler", "Subscribe to our newsletter for weekly updates"),
    ("boiler", "Terms of Use  Privacy Policy  Contact us"),
    ("boiler", "We process your data to deliver content or advertisements"),
    ("boiler", "Sign in to continue reading this article"),
]

CRITERIA = {
    "ai": "This is about AI, machine learning, or computer chips.",
    "food": "This is about cooking, restaurants, or food.",
    "sport": "This is about sports or athletes.",
    "finance": "This is about money, markets, or business finance.",
    "nature": "This is about animals or the natural world.",
    "health": "This is about health, medicine, or the human body.",
}


def build_content(text, criterion):
    return (f'State / Context:\nElement: <a>\nText: "{text}"\n\n'
            f'Question:\nThe user is looking for: {criterion}\n\n'
            f'Does this page element match what the user is looking for?\n\n'
            f'Answer with Yes or No only.')


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a, b in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def fit_probe(X, y, ridge=1.0):
    """Least squares to +-1 targets; a logistic fit gives the same ranking."""
    A = np.c_[X, np.ones(len(X))]
    target = y * 2.0 - 1.0
    reg = ridge * np.eye(A.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(A.T @ A + reg, A.T @ target)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", nargs="+", default=["nature", "health"],
                    help="criteria the probe is never trained on")
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from mini_jev.engine_jax import MiniJevEngine

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    model.eval()
    engine = MiniJevEngine(model_id=BASE, warmup=False)

    yes = [tok.encode(s, add_special_tokens=False)[0] for s in ("Yes", "yes")]
    no = [tok.encode(s, add_special_tokens=False)[0] for s in ("No", "no")]

    rows = []
    total = len(ITEMS) * len(CRITERIA)
    print(f"{total} (element, criterion) pairs -- one forward pass each")
    for ci, (ckey, criterion) in enumerate(CRITERIA.items()):
        for topic, text in ITEMS:
            ids = tok.encode(engine._build_prompt(build_content(text, criterion)),
                             add_special_tokens=False)
            with torch.no_grad():
                out = model(torch.tensor([ids]), output_hidden_states=True)
            logits = out.logits[0, -1].float().numpy()
            a = max(logits[yes[0]], logits[yes[1]])
            b = max(logits[no[0]], logits[no[1]])
            rows.append({
                "criterion": ckey,
                "label": 1 if topic == ckey else 0,
                "p": float(np.exp(a) / (np.exp(a) + np.exp(b))),
                "h": out.hidden_states[-1][0, -1].float().numpy(),
            })
        print(f"  {ckey:8s} done ({(ci + 1) * len(ITEMS)}/{total})")

    H = np.array([r["h"] for r in rows])
    y = np.array([r["label"] for r in rows])
    p = np.array([r["p"] for r in rows])
    crit = np.array([r["criterion"] for r in rows])

    mu, sd = H.mean(0), H.std(0) + 1e-6
    Hn = (H - mu) / sd

    print(f"\n[1] Yes/No logit read, all {len(rows)} pairs")
    print(f"    p range {p.min():.4f}-{p.max():.4f}   AUC {auc(p, y):.3f}")

    held = set(args.holdout)
    tr = np.array([c not in held for c in crit])
    te = ~tr

    w = fit_probe(Hn[tr], y[tr])
    s_te = np.c_[Hn[te], np.ones(te.sum())] @ w

    print(f"\n[2] Linear probe on the hidden state")
    print(f"    trained on {sorted(set(crit[tr]))}")
    print(f"    tested on  {sorted(set(crit[te]))}  (never seen)")
    print(f"    AUC {auc(s_te, y[te]):.3f}   "
          f"accuracy {(np.sign(s_te) > 0).astype(int).__eq__(y[te]).mean():.0%}")

    print(f"\n[3] Per held-out criterion")
    for c in sorted(held):
        m = crit == c
        sc = np.c_[Hn[m], np.ones(m.sum())] @ w
        print(f"    {c:8s} AUC {auc(sc, y[m]):.3f}   "
              f"p(Yes/No) AUC {auc(p[m], y[m]):.3f}   "
              f"({y[m].sum()} relevant of {m.sum()})")


if __name__ == "__main__":
    raise SystemExit(main())
