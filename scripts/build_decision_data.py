"""
Build typed decision requests for LoRA training.

The first probe scored 0.977 on held-out criteria and 0.481 on a real page.
The cause was the training data, not the model: its negatives were cookie
banners and newsletter prompts, so the probe learned "has content" rather than
"is on topic", and on a page where every item is a paper title that axis is
flat.

So the rule here is that a negative is always *the same kind of text about a
different subject*, drawn from the same corpus as its positive. That is the
discrimination a page filter actually has to make.

Paper and abstract corpora are deliberately excluded. The evaluation is
alphaXiv paper titles, and it is only worth anything while it stays out of
domain.

    python3 scripts/build_decision_data.py --out data/decision
"""

import argparse
import json
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Several phrasings, so the head learns the task rather than one sentence.
CRITERION_TEMPLATES = [
    "This is about {}.",
    "The user is looking for: {}",
    "Is this relevant to someone interested in {}?",
    "I want to see things about {}.",
    "Show me content about {}.",
    "{}",
]
NOUL_INSTRUCTIONS = [
    "Does this match what the user is looking for?",
    "Is this relevant?",
    "Does this item match the criterion?",
]


def humanize(name):
    name = name.replace("_", " ").replace(".", " ").replace("-", " ")
    name = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name)
    return " ".join(name.split()).lower()


def clip(text, n=700):
    text = " ".join(str(text).split())
    return text[:n]


def corpora(seed, n_doc):
    """(name, [(text, class_name)], ordered_levels_or_None) per corpus."""
    from datasets import load_dataset
    out = []

    def take(ds, n):
        return ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    d = take(load_dataset("fancyzhx/ag_news", split="train"), n_doc)
    names = ["world news", "sports", "business", "science and technology"]
    out.append(("ag_news", [(clip(r["text"]), names[r["label"]]) for r in d], None))

    d = take(load_dataset("fancyzhx/dbpedia_14", split="train"), n_doc)
    names = [humanize(x) for x in
             ["Company", "EducationalInstitution", "Artist", "Athlete",
              "OfficeHolder", "MeanOfTransportation", "Building",
              "NaturalPlace", "Village", "Animal", "Plant", "Album",
              "Film", "WrittenWork"]]
    out.append(("dbpedia", [(clip(f'{r["title"]}. {r["content"]}'),
                             names[r["label"]]) for r in d], None))

    d = take(load_dataset("community-datasets/yahoo_answers_topics", split="train"), n_doc)
    names = ["society and culture", "science and mathematics", "health",
             "education and reference", "computers and internet", "sports",
             "business and finance", "entertainment and music",
             "family and relationships", "politics and government"]
    out.append(("yahoo", [(clip(f'{r["question_title"]} {r["question_content"]}'),
                           names[r["topic"]]) for r in d], None))

    d = take(load_dataset("SetFit/20_newsgroups", split="train"), n_doc)
    out.append(("newsgroups", [(clip(r["text"]), humanize(r["label_text"]))
                               for r in d if r["text"].strip()], None))

    d = take(load_dataset("clinc/clinc_oos", "small", split="train"), n_doc)
    names = d.features["intent"].names
    out.append(("clinc", [(clip(r["text"]), humanize(names[r["intent"]]))
                          for r in d], None))

    d = take(load_dataset("legacy-datasets/banking77", split="train"), n_doc)
    names = d.features["label"].names
    out.append(("banking77", [(clip(r["text"]), humanize(names[r["label"]]))
                              for r in d], None))

    d = take(load_dataset("dair-ai/emotion", split="train"), n_doc)
    names = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    out.append(("emotion", [(clip(r["text"]), names[r["label"]]) for r in d], None))

    # Ordinal corpora drive `score`, where the options have an order.
    d = take(load_dataset("SetFit/sst5", split="train"), n_doc // 2)
    lv = ["very negative", "negative", "neutral", "positive", "very positive"]
    out.append(("sst5", [(clip(r["text"]), lv[r["label"]]) for r in d], lv))

    d = take(load_dataset("Yelp/yelp_review_full", split="train"), n_doc // 2)
    lv = ["one star", "two stars", "three stars", "four stars", "five stars"]
    out.append(("yelp", [(clip(r["text"], 500), lv[r["label"]]) for r in d], lv))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "decision"))
    ap.add_argument("--n-doc", type=int, default=6000,
                    help="documents sampled per corpus")
    ap.add_argument("--pos-rate", type=float, default=0.35,
                    help="share of noul items whose criterion matches")
    ap.add_argument("--max-options", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)

    recs = []
    for name, docs, levels in corpora(args.seed, args.n_doc):
        classes = sorted({c for _, c in docs})
        print(f"{name:12s} {len(docs):6d} docs · {len(classes)} classes")

        for text, cls in docs:
            kind = rng.random()

            # The core format: a criterion in natural language, Yes/No.
            if kind < 0.6:
                match = rng.random() < args.pos_rate
                # A negative names a different class of the SAME corpus, so the
                # model cannot win by noticing what kind of text this is.
                topic = cls if match else rng.choice([c for c in classes if c != cls])
                crit = rng.choice(CRITERION_TEMPLATES).format(topic)
                recs.append({
                    "src": f"{name}:noul", "type": "noul", "state": text,
                    "instructions": f"{crit}\n\n{rng.choice(NOUL_INSTRUCTIONS)}",
                    "options": ["Yes", "No"], "gold": 0 if match else 1,
                })

            # Choice over the corpus's own labels.
            elif kind < 0.9:
                others = [c for c in classes if c != cls]
                k = min(args.max_options - 1, len(others))
                opts = rng.sample(others, k) + [cls]
                rng.shuffle(opts)
                recs.append({
                    "src": f"{name}:choice", "type": "choice", "state": text,
                    "instructions": "Which of these describes it best?",
                    "options": opts, "gold": opts.index(cls),
                })

            # Score, only where the labels are genuinely ordered.
            elif levels:
                recs.append({
                    "src": f"{name}:score", "type": "score", "state": text,
                    "instructions": "Rate it on this scale.",
                    "options": levels, "gold": levels.index(cls),
                })

    rng.shuffle(recs)
    n_dev = len(recs) // 20
    splits = {"dev": recs[:n_dev], "train": recs[n_dev:]}
    for k, v in splits.items():
        p = os.path.join(args.out, f"{k}.jsonl")
        with open(p, "w") as f:
            for r in v:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{k:6s} {len(v):7,d} -> {p}")

    by_type = {}
    for r in recs:
        by_type[r["type"]] = by_type.get(r["type"], 0) + 1
    print("\nby type:", by_type)
    noul = [r for r in recs if r["type"] == "noul"]
    print(f"noul positive rate {sum(1 for r in noul if r['gold'] == 0) / len(noul):.1%}")
    print(f"largest option count: {max(len(r['options']) for r in recs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
