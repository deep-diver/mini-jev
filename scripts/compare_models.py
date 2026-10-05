"""
Compare Gemma 3 270M against Gemma 4 E2B on the SurfMate region-type task.

Ground truth comes from the markup itself: a <nav>, <main>, <footer> or an
explicit ARIA role states what a region is. Those nodes are collected, the
declaring tag and role are then **masked out**, and each model has to recover
the answer from what is left -- size, position, contents and text. That is
precisely the production case, since the server only asks the model about
regions whose markup says nothing, and it gives real labels without the
circularity of scoring a model against another model's opinion.

    python3 scripts/compare_models.py --limit 50
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from surfmate_server import (  # noqa: E402
    PAGE_QUESTIONS, ROLE_ALIASES, describe_container, dom_role,
)

SURFMATE = os.path.expanduser("~/Developers/SurfMate")
SNAPSHOTS = os.path.join(ROOT, "data", "surfmate", "snapshots.jsonl")

PAGES = [
    "https://www.python.org/",
    "https://en.wikipedia.org/wiki/Web_accessibility",
    "https://github.com/explore",
    "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API",
    "https://www.bbc.com/news",
    "https://books.toscrape.com/",
    "https://www.gov.uk/",
    "https://www.nasa.gov/",
]

CHROME_SHIM = """
window.chrome = {
  runtime: { onMessage: { addListener: () => {} }, sendMessage: () => {},
             getURL: (p) => p, lastError: null },
  storage: { local: { get: (k, cb) => cb && cb({}), set: (o, cb) => cb && cb() },
             onChanged: { addListener: () => {} } },
};
"""


def collect_snapshots(path):
    """Run the extension's own snapshot builder over real pages and cache it."""
    from playwright.sync_api import sync_playwright

    with open(os.path.join(SURFMATE, "content.js")) as f:
        content_js = f.read()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        with open(path, "w") as out:
            for url in PAGES:
                page = ctx.new_page()
                try:
                    page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    page.wait_for_timeout(1800)
                    page.evaluate(CHROME_SHIM)
                    page.evaluate(content_js)
                    snap = page.evaluate("generateDOMSnapshot()")
                    out.write(json.dumps(snap, ensure_ascii=False) + "\n")
                    print(f"  {len(snap['elements']):4d} elements  {url}")
                except Exception as e:
                    print(f"  FAILED {url}: {str(e).splitlines()[0][:80]}")
                finally:
                    page.close()
        browser.close()


def mask(el):
    """Strip the markup that gives the answer away."""
    masked = dict(el)
    masked["tag"] = "div"
    attrs = dict(el.get("attributes") or {})
    attrs["role"] = None
    masked["attributes"] = attrs
    return masked


def build_set(limit, seed=0):
    items = []
    with open(SNAPSHOTS) as f:
        for line in f:
            snap = json.loads(line)
            title = snap.get("title", "")
            for el in snap.get("elements", []):
                if not el.get("isContainer"):
                    continue
                truth = dom_role(el)
                if truth is None:
                    continue
                items.append({
                    "truth": truth,
                    "state": describe_container(mask(el), title),
                    "url": snap.get("url", ""),
                })

    # Balance the classes a little: <nav> vastly outnumbers everything else, and
    # a model that always answers "navigation" would otherwise look good.
    random.Random(seed).shuffle(items)
    per_class, capped = {}, []
    cap = max(3, limit // 4)
    for it in items:
        if per_class.get(it["truth"], 0) >= cap:
            continue
        per_class[it["truth"]] = per_class.get(it["truth"], 0) + 1
        capped.append(it)
    return capped[:limit]


def run(model, items, use_batch=True):
    from mini_jev import MiniJevClient

    t0 = time.time()
    client = MiniJevClient(model=model)
    load_s = time.time() - t0

    states = [it["state"] for it in items]
    t0 = time.time()
    if use_batch:
        responses = client.evaluate_many(states, PAGE_QUESTIONS)
    else:
        responses = [client.evaluate(s, PAGE_QUESTIONS) for s in states]
    elapsed = time.time() - t0

    preds, nouls, confs = [], [], []
    for res in responses:
        guess = res.answers["role"]
        preds.append(ROLE_ALIASES.get(guess.choice, guess.choice))
        confs.append(guess.confidence)
        nouls.append(res.answers["is_container"].noul)

    correct = sum(p == it["truth"] for p, it in zip(preds, items))
    return {
        "model": model,
        "load_s": round(load_s, 1),
        "total_s": round(elapsed, 1),
        "ms_per_node": round(elapsed / len(items) * 1000, 1),
        "accuracy": correct / len(items),
        "correct": correct,
        "n": len(items),
        "distinct_roles_used": len(set(preds)),
        "pred_counts": {r: preds.count(r) for r in sorted(set(preds))},
        "mean_confidence": round(statistics.mean(confs), 3),
        "noul_stdev": round(statistics.pstdev(nouls), 4),
        "noul_mean": round(statistics.mean(nouls), 3),
        "preds": preds,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--refresh", action="store_true", help="re-collect page snapshots")
    ap.add_argument("--models", nargs="+",
                    default=["google/gemma-3-270m-it", "google/gemma-4-E2B-it"])
    args = ap.parse_args()

    if args.refresh or not os.path.exists(SNAPSHOTS):
        print("collecting snapshots...")
        collect_snapshots(SNAPSHOTS)

    items = build_set(args.limit)
    truth_counts = {}
    for it in items:
        truth_counts[it["truth"]] = truth_counts.get(it["truth"], 0) + 1
    print(f"\n{len(items)} labelled nodes (tag and role masked out)")
    print(f"ground truth: {truth_counts}")
    print(f"majority-class baseline: {max(truth_counts.values()) / len(items):.1%}\n")

    results = []
    for model in args.models:
        print(f"--- {model} (batched)")
        res = run(model, items)
        results.append(res)
        print(f"    accuracy {res['accuracy']:.1%} ({res['correct']}/{res['n']})  "
              f"{res['ms_per_node']} ms/node  "
              f"{res['distinct_roles_used']}/9 role classes used")
        print(f"    predictions: {res['pred_counts']}")
        print(f"    mean role confidence {res['mean_confidence']}, "
              f"p(container) mean {res['noul_mean']} stdev {res['noul_stdev']}\n")

    out = os.path.join(ROOT, "data", "surfmate", "model_comparison.json")
    with open(out, "w") as f:
        json.dump({"truth_counts": truth_counts, "results": results}, f, indent=2)
    print(f"-> {out}")

    if len(results) == 2:
        a, b = results
        print(f"\n{b['model']} vs {a['model']}: "
              f"accuracy {a['accuracy']:.1%} -> {b['accuracy']:.1%}, "
              f"{b['ms_per_node'] / max(a['ms_per_node'], 1e-9):.0f}x the cost per node")


if __name__ == "__main__":
    raise SystemExit(main())
