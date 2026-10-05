"""
Build a labelled dataset for the region-type task.

Labels come from the markup: a <nav>, <main>, <footer>, <aside>, <form> or an
explicit ARIA role states what a region is. Those nodes are collected and the
declaring tag and role are then masked out, so the model has to recover the
answer from size, position, contents and text -- which is exactly the
production case, since the server only consults the model about regions whose
markup says nothing.

The split is by PAGE, never by node. Nodes from one page share a template, a
class vocabulary and a layout, so a node-level split would put near-duplicates
on both sides and report a score the model has not earned.

    python3 scripts/build_role_dataset.py --harvest    # visit the sites first
    python3 scripts/build_role_dataset.py              # rebuild from cache
"""

import argparse
import json
import os
import random
import sys

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from surfmate_server import describe_container, dom_role  # noqa: E402
from survey_pages import SITES  # noqa: E402

SURFMATE = os.path.expanduser("~/Developers/SurfMate")
DATA_DIR = os.path.join(ROOT, "data", "surfmate")
SNAPSHOTS = os.path.join(DATA_DIR, "role_snapshots.jsonl")

CHROME_SHIM = """
window.chrome = {
  runtime: { onMessage: { addListener: () => {} }, sendMessage: () => {},
             getURL: (p) => p, lastError: null },
  storage: { local: { get: (k, cb) => cb && cb({}), set: (o, cb) => cb && cb() },
             onChanged: { addListener: () => {} } },
};
"""

# Pages that serve a bot challenge produce an interstitial rather than the site,
# and its handful of nodes would be labelled as if they were real content.
BLOCKED_TITLES = ("just a moment", "access denied", "403 forbidden",
                  "client challenge", "attention required", "are you a robot")


def harvest(path):
    from playwright.sync_api import sync_playwright

    with open(os.path.join(SURFMATE, "content.js")) as f:
        content_js = f.read()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    kept = 0
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"))
        with open(path, "w") as out:
            for category, name, url in SITES:
                page = ctx.new_page()
                try:
                    page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    page.wait_for_timeout(1800)
                    page.evaluate(CHROME_SHIM)
                    page.evaluate(content_js)
                    snap = page.evaluate("generateDOMSnapshot()")
                    title = (snap.get("title") or "").lower()
                    if any(b in title for b in BLOCKED_TITLES):
                        print(f"  {name:16s} BLOCKED ({snap.get('title','')[:28]})")
                        continue
                    labelled = sum(1 for e in snap.get("elements", [])
                                   if e.get("isContainer") and dom_role(e))
                    if labelled == 0:
                        print(f"  {name:16s} no labelled nodes")
                        continue
                    snap["site"] = name
                    snap["category"] = category
                    out.write(json.dumps(snap, ensure_ascii=False) + "\n")
                    kept += 1
                    print(f"  {name:16s} {labelled:3d} labelled nodes")
                except Exception as e:
                    print(f"  {name:16s} FAILED {str(e).splitlines()[0][:50]}")
                finally:
                    page.close()
        browser.close()
    print(f"\n{kept} pages -> {path}")


def mask(el):
    """Strip the markup that gives the answer away."""
    masked = dict(el)
    masked["tag"] = "div"
    attrs = dict(el.get("attributes") or {})
    attrs["role"] = None
    masked["attributes"] = attrs
    return masked


def build(seed=0, test_frac=0.25):
    pages = []
    with open(SNAPSHOTS) as f:
        for line in f:
            snap = json.loads(line)
            items = []
            for el in snap.get("elements", []):
                if not el.get("isContainer"):
                    continue
                truth = dom_role(el)
                if truth is None:
                    continue
                items.append({
                    "truth": truth,
                    "state": describe_container(mask(el), snap.get("title", "")),
                    "site": snap.get("site"),
                    "category": snap.get("category"),
                })
            if items:
                pages.append(items)

    rng = random.Random(seed)
    rng.shuffle(pages)
    n_test = max(1, int(len(pages) * test_frac))
    test = [it for pg in pages[:n_test] for it in pg]
    train = [it for pg in pages[n_test:] for it in pg]

    # Within a page the same region often appears at several nesting levels with
    # near-identical serialisations; keep one of each so the loss is not
    # dominated by whichever page had the deepest wrappers.
    def dedupe(rows):
        seen, out = set(), []
        for r in rows:
            key = (r["truth"], r["state"])
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    return dedupe(train), dedupe(test), len(pages) - n_test, n_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", action="store_true", help="re-visit the sites")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.harvest or not os.path.exists(SNAPSHOTS):
        print("harvesting...")
        harvest(SNAPSHOTS)

    train, test, n_tr_pages, n_te_pages = build(seed=args.seed)

    for name, rows in (("train", train), ("test", test)):
        counts = {}
        for r in rows:
            counts[r["truth"]] = counts.get(r["truth"], 0) + 1
        majority = max(counts.values()) / len(rows) if rows else 0
        print(f"{name:5s} {len(rows):5d} nodes  majority-class {majority:.1%}  {counts}")

        with open(os.path.join(DATA_DIR, f"role_{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nsplit by page: {n_tr_pages} train pages / {n_te_pages} test pages")
    print(f"test sites: {sorted({r['site'] for r in test})}")
    print(f"-> {DATA_DIR}/role_train.jsonl, role_test.jsonl")


if __name__ == "__main__":
    raise SystemExit(main())
