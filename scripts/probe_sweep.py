"""
Find a readout of Gemma 3 270M that actually works for runtime criteria.

The Yes/No token logits carry almost nothing (AUC ~0.585): every answer comes
back at ~0.999. The hidden states of the same forward pass do carry the answer,
so this sweeps the readout rather than the model: which layer, what centring,
linear or not, and how much regularisation.

Evaluation is leave-one-criterion-out. The criterion is supplied by the user at
call time, so the only number that matters is performance on a criterion the
probe was never fitted on; scoring on a criterion it has seen would measure
memorisation of that topic instead.

    python3 scripts/probe_sweep.py --extract      # forward passes, cached
    python3 scripts/probe_sweep.py                # sweep the cache
"""

import argparse
import itertools
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

BASE = "google/gemma-3-270m-it"
CACHE = os.path.join(ROOT, "data", "surfmate", "probe_states.npz")

# Sixteen topics, so a probe can be fitted on fifteen and tested on the
# sixteenth. Items are deliberately plain page text -- headlines, list rows,
# boilerplate -- because that is what a page filter actually sees.
TOPICS = {
    "ai": ["Nvidia unveils new Blackwell GPU architecture for AI training",
           "How large language models are changing software engineering",
           "OpenAI releases a smaller, faster reasoning model",
           "Quantization brings a 70B model onto a laptop",
           "Benchmarking int8 inference on Apple silicon"],
    "food": ["Why chef Ferran Adria is the subject of a new documentary",
             "A history of sourdough baking in northern Europe",
             "The best ramen shops in the city, ranked",
             "How to braise short ribs without drying them out",
             "Fermentation basics: kimchi, miso and vinegar"],
    "sport": ["Manchester City beat Arsenal in a late comeback",
              "The marathon world record falls again in Berlin",
              "How a rookie point guard changed the season",
              "Olympic swimming trials: who qualified overnight",
              "Cycling team announces its Tour de France roster"],
    "finance": ["Treasury yields climb as inflation data surprises",
                "A startup raises $40M in a round led by Sequoia",
                "Why the central bank held rates steady this month",
                "Quarterly earnings beat expectations on cloud revenue",
                "Retail investors pile into a newly listed company"],
    "nature": ["Weeping whales: stillborn humpback whale grieving documented",
               "T. rex teeth indicate it ran as warm as an elephant",
               "Bear-y good neighbours: humans and animals share a town",
               "A coral reef recovers faster than anyone predicted",
               "Migrating birds navigate using a magnetic sense"],
    "health": ["Finding the cells that put our brain to sleep",
               "A new trial for a once-weekly diabetes drug",
               "Why sleep debt cannot simply be repaid at the weekend",
               "Physiotherapy beats surgery for some knee injuries",
               "Vitamin D supplements show no effect in a large study"],
    "travel": ["Forty-eight hours in Lisbon on a small budget",
               "The sleeper train returning to central Europe",
               "How to avoid the crowds at the summer festivals",
               "A walking route along the old pilgrim path",
               "Visa rules change for long-stay travellers"],
    "auto": ["Don't call it an SUV: the Ferrari Purosangue review",
             "An electric hatchback that finally undercuts petrol",
             "Why solid-state batteries keep slipping a year",
             "Used car prices fall for the third quarter running",
             "A hands-on with the new hybrid drivetrain"],
    "music": ["A jazz quartet reunites after thirty years apart",
              "The album that defined a decade of British pop",
              "Vinyl sales overtake CDs for the first time since 1987",
              "How a bedroom producer topped the streaming chart",
              "Orchestra announces its season of late romantics"],
    "film": ["A quiet drama wins the top prize at the festival",
             "The sequel nobody asked for is surprisingly good",
             "How practical effects made a comeback this year",
             "Director discusses the ten-year road to release",
             "Box office returns to pre-pandemic levels"],
    "politics": ["Parliament votes down the housing amendment",
                 "A coalition forms after three weeks of talks",
                 "The mayor announces a transit funding plan",
                 "Election turnout rises in rural districts",
                 "New rules on lobbying take effect in January"],
    "education": ["Universities rethink the lecture after a decade of decline",
                  "A school district pilots a four-day week",
                  "Tuition freezes extended for another year",
                  "How apprenticeships are filling the skills gap",
                  "Reading scores recover slowly after the pandemic"],
    "fashion": ["Readers on their favourite second-hand fashion finds",
                "The return of tailoring on the autumn runway",
                "Why the sneaker resale market finally cooled",
                "A label built entirely on deadstock fabric",
                "Fashion week moves to a smaller venue"],
    "gaming": ["An indie roguelike tops the charts on launch day",
               "The studio delays its open-world title again",
               "Handheld consoles are having a second moment",
               "Speedrunners break a record thought unbeatable",
               "A remaster that respects the original's pacing"],
    "space": ["Rings around a tiny body have changed over the past decade",
              "The lander touches down near the lunar south pole",
              "A telescope spots the earliest galaxy yet",
              "Launch cadence doubles at the coastal pad",
              "Astronauts complete a six-hour spacewalk"],
    "boiler": ["Subscribe to our newsletter for weekly updates",
               "Terms of Use  Privacy Policy  Contact us",
               "We process your data to deliver content or advertisements",
               "Sign in to continue reading this article",
               "Accept all cookies to continue browsing"],
}

CRITERIA = {
    "ai": "This is about AI, machine learning, or computer chips.",
    "food": "This is about cooking, restaurants, or food.",
    "sport": "This is about sports or athletes.",
    "finance": "This is about money, markets, or business finance.",
    "nature": "This is about animals or the natural world.",
    "health": "This is about health, medicine, or the human body.",
    "travel": "This is about travel or places to visit.",
    "auto": "This is about cars or vehicles.",
    "music": "This is about music or musicians.",
    "film": "This is about films or cinema.",
    "politics": "This is about politics or government.",
    "education": "This is about schools, universities, or education.",
    "fashion": "This is about fashion or clothing.",
    "gaming": "This is about video games.",
    "space": "This is about space or astronomy.",
}


def build_content(text, criterion):
    return (f'State / Context:\nElement: <a>\nText: "{text}"\n\n'
            f'Question:\nThe user is looking for: {criterion}\n\n'
            f'Does this page element match what the user is looking for?\n\n'
            f'Answer with Yes or No only.')


def extract(path):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from mini_jev.engine_jax import MiniJevEngine

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    model.eval()
    engine = MiniJevEngine(model_id=BASE, warmup=False)

    yes = [tok.encode(s, add_special_tokens=False)[0] for s in ("Yes", "yes")]
    no = [tok.encode(s, add_special_tokens=False)[0] for s in ("No", "no")]

    items = [(topic, text) for topic, texts in TOPICS.items() for text in texts]
    total = len(items) * len(CRITERIA)
    print(f"{len(items)} items x {len(CRITERIA)} criteria = {total} forward passes")

    states, labels, crits, ps = [], [], [], []
    done = 0
    for ckey, criterion in CRITERIA.items():
        for topic, text in items:
            ids = tok.encode(engine._build_prompt(build_content(text, criterion)),
                             add_special_tokens=False)
            with torch.no_grad():
                out = model(torch.tensor([ids]), output_hidden_states=True)
            logits = out.logits[0, -1].float().numpy()
            a = max(logits[yes[0]], logits[yes[1]])
            b = max(logits[no[0]], logits[no[1]])
            ps.append(float(np.exp(a) / (np.exp(a) + np.exp(b))))
            # Every layer, taken at the answer position.
            states.append(np.stack([h[0, -1].float().numpy() for h in out.hidden_states]))
            labels.append(1 if topic == ckey else 0)
            crits.append(ckey)
            done += 1
        print(f"  {ckey:10s} {done}/{total}")

    np.savez_compressed(path, states=np.array(states, dtype=np.float32),
                        labels=np.array(labels), crits=np.array(crits),
                        ps=np.array(ps))
    print(f"-> {path}")


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    return sum((a > b) + 0.5 * (a == b)
               for a, b in itertools.product(pos, neg)) / (len(pos) * len(neg))


def fit(X, y, ridge):
    A = np.c_[X, np.ones(len(X))]
    R = ridge * np.eye(A.shape[1])
    R[-1, -1] = 0.0
    return np.linalg.solve(A.T @ A + R, A.T @ (y * 2.0 - 1.0))


def prepare(H, crits, centre):
    """Feature prep. Within-batch centring is available for free at inference:
    a page filter always scores every element on the page at once."""
    X = H.astype(np.float64).copy()
    if centre == "batch":
        for c in set(crits):
            m = crits == c
            X[m] -= X[m].mean(0)
    elif centre == "global":
        X -= X.mean(0)
    return X / (X.std(0) + 1e-6)


def loco(H, y, crits, ridge, centre):
    """Leave one criterion out: fit on the rest, score the held-out one."""
    X = prepare(H, crits, centre)
    scores = np.zeros(len(y))
    for c in sorted(set(crits)):
        te = crits == c
        w = fit(X[~te], y[~te], ridge)
        scores[te] = np.c_[X[te], np.ones(te.sum())] @ w
    per = {c: auc(scores[crits == c], y[crits == c]) for c in sorted(set(crits))}
    return float(np.mean(list(per.values()))), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()

    if args.extract or not os.path.exists(CACHE):
        extract(CACHE)

    d = np.load(CACHE, allow_pickle=True)
    S, y, crits, ps = d["states"], d["labels"], d["crits"], d["ps"]
    n_layers = S.shape[1]
    print(f"\ncached: {S.shape[0]} pairs, {n_layers} layers, {S.shape[2]} dims")
    print(f"Yes/No logit read: AUC {auc(ps, y):.3f}  "
          f"(p range {ps.min():.4f}-{ps.max():.4f})\n")

    print(f"{'layer':>6} {'global':>9} {'batch':>9}")
    best = (0, None)
    for li in range(n_layers):
        row = {}
        for centre in ("global", "batch"):
            scores = [loco(S[:, li], y, crits, r, centre)[0] for r in (1, 10, 100)]
            row[centre] = max(scores)
            if row[centre] > best[0]:
                best = (row[centre], (li, centre))
        print(f"{li:>6} {row['global']:>9.3f} {row['batch']:>9.3f}")

    li, centre = best[1]
    print(f"\nbest linear readout: layer {li}, {centre} centring, "
          f"mean LOCO AUC {best[0]:.3f}")
    for r in (1, 10, 100, 300):
        m, per = loco(S[:, li], y, crits, r, centre)
        worst = min(per.items(), key=lambda kv: kv[1])
        print(f"  ridge {r:>4}: mean {m:.3f}   worst {worst[0]} {worst[1]:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
