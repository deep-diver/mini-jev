"""
Measure the relevance probe against hand labels on real alphaXiv titles.

The cruise reports "26 found, 97 skipped" and nothing about whether those were
the right 26. This fits nothing and tunes nothing: it scores harvested titles
through the running server and compares to labels written before the scores
were seen.

    python3 scripts/surfmate_server.py     # in one shell
    python3 scripts/eval_alphaxiv.py       # in another
"""

import json
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TITLES = os.path.join(ROOT, "data", "eval", "alphaxiv_titles.json")
LABELS = os.path.join(ROOT, "data", "eval", "alphaxiv_labels.json")
CRITERION = "This is about reinforcement learning, agents, or training with rewards."

# Positive = the title names reinforcement learning / rewards, or an autonomous
# agent. Borderline cases went the conservative way: "on-policy distillation"
# has no reward, a "VLA policy" is not by itself RL.
POSITIVE = {
    3, 7, 12, 17, 27, 28, 31, 36, 37, 39, 40, 41, 43, 53, 54, 57, 63, 71, 75,
    82, 85, 86, 96, 97, 105, 111, 113, 124, 125, 131, 139, 140, 145, 147, 149,
    152, 158, 161, 167, 169,
}


def auc(scores, y):
    pos = [s for s, t in zip(scores, y) if t]
    neg = [s for s, t in zip(scores, y) if not t]
    if not pos or not neg:
        return float("nan")
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def main():
    items = json.load(open(TITLES))
    y = [i in POSITIVE for i in range(len(items))]

    body = json.dumps({
        "criterion": CRITERION,
        "elements": [{"id": i, "text": it["title"]} for i, it in enumerate(items)],
        "session": "eval", "reset": True,
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:8777/filter_feed", body,
                                 {"content-type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=900))
    if not d.get("scores"):
        print("server returned no scores:", d)
        return 1
    s = d["scores"]

    print(f"{len(items)} titles · {sum(y)} positive ({sum(y) / len(y):.0%}) · "
          f"{d['elapsed_ms']} ms ({d['per_decision_ms']} ms each)\n")
    print(f"ranking quality: AUC {auc(s, y):.3f}\n")

    print(f"{'thresh':>7} {'kept':>5} {'prec':>7} {'recall':>7} {'F1':>6}")
    for th in (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60):
        tp = sum(1 for a, t in zip(s, y) if a >= th and t)
        fp = sum(1 for a, t in zip(s, y) if a >= th and not t)
        fn = sum(1 for a, t in zip(s, y) if a < th and t)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        mark = "  <- in use" if abs(th - 0.30) < 1e-9 else ""
        print(f"{th:7.2f} {tp + fp:5d} {p:7.1%} {r:7.1%} {f1:6.2f}{mark}")

    ranked = sorted(zip(s, y, [it["title"] for it in items]), reverse=True)
    print("\ntop 15 (O = correct per the hand labels):")
    for a, t, title in ranked[:15]:
        print(f"   {a:.2f}  {'O' if t else 'X'}  {title[:68]}")
    print("\nthe 8 lowest-scoring positives it missed:")
    for a, t, title in sorted((x for x in ranked if x[1]))[:8]:
        print(f"   {a:.2f}     {title[:68]}")

    json.dump([{"i": i, "title": items[i]["title"], "label": int(y[i]),
                "score": s[i]} for i in range(len(items))],
              open(LABELS, "w"), ensure_ascii=False, indent=1)
    print(f"\nlabels and scores -> {LABELS}  (all of them, so the run can be audited)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
