"""
Run the untuned model over harvested SurfMate candidates.

This is the "before" half of the comparison: base Gemma 3 270M, no tuning,
same prompts the tuned model will later see.

    python3 scripts/run_baseline.py --limit 20      # quick smoke pass
    python3 scripts/run_baseline.py                 # full harvest
"""

import argparse
import json
import os
import statistics
import sys
import time

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from surfmate_task import QUESTIONS, format_node  # noqa: E402


def load_nodes(path, limit=None, per_page=None):
    items = []
    with open(path) as f:
        for ln in f:
            page = json.loads(ln)
            nodes = page["nodes"]
            if per_page and len(nodes) > per_page:
                # Nodes are area-sorted, so taking a prefix would sample nothing
                # but page-sized wrappers. Spread the picks across the ranking.
                step = len(nodes) / per_page
                nodes = [nodes[int(i * step)] for i in range(per_page)]
            for node in nodes:
                items.append({"page": page, "node": node})
    if limit:
        items = items[:limit]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=os.path.join(ROOT, "data", "surfmate", "pages.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "surfmate", "baseline.jsonl"))
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--limit", type=int, help="cap total nodes")
    ap.add_argument("--per-page", type=int, help="cap nodes per page")
    args = ap.parse_args()

    items = load_nodes(args.pages, args.limit, args.per_page)
    print(f"{len(items)} nodes from {args.pages}")

    from mini_jev import MiniJevClient

    t0 = time.time()
    client = MiniJevClient(model=args.model)
    print(f"model ready in {time.time() - t0:.1f}s on {client.devices}")

    prompts = [format_node(it["node"], it["page"]) for it in items]

    tok = len(prompts[0].split())
    print(f"\n--- sample prompt (~{tok} words) ---\n{prompts[0]}\n---\n")

    t0 = time.time()
    responses = client.evaluate_batch(states=prompts, questions=QUESTIONS)
    elapsed = time.time() - t0

    lat = [r.usage.latency_ms for r in responses]
    intok = [r.usage.input_tokens for r in responses]
    print(
        f"{len(responses)} nodes in {elapsed:.1f}s "
        f"({elapsed / len(responses) * 1000:.1f} ms/node, "
        f"{len(responses) / elapsed:.1f} nodes/sec)"
    )
    print(
        f"per-node latency: mean {statistics.mean(lat):.1f} ms, "
        f"p50 {statistics.median(lat):.1f} ms, max {max(lat):.1f} ms"
    )
    print(f"input tokens: mean {statistics.mean(intok):.0f}, max {max(intok)}")

    with open(args.out, "w") as out:
        for it, resp in zip(items, responses):
            out.write(json.dumps({
                "url": it["page"]["url"],
                "node_id": it["node"]["id"],
                "selector": it["node"]["selector"],
                "tag": it["node"]["tag"],
                "is_container": resp.answers["is_container"].noul,
                "role": resp.answers["role"].choice,
                "role_probs": resp.answers["role"].probabilities,
                "importance": resp.answers["importance"].score,
                "latency_ms": resp.usage.latency_ms,
            }, ensure_ascii=False) + "\n")
    print(f"-> {args.out}")

    # The failure mode to look for is not "wrong" but "constant": a base model
    # that answers the same thing everywhere carries no signal to tune away from.
    nouls = [r.answers["is_container"].noul for r in responses]
    roles = [r.answers["role"].choice for r in responses]
    scores = [r.answers["importance"].score for r in responses]

    print("\n--- base model output distribution ---")
    print(f"is_container p(Yes): mean {statistics.mean(nouls):.3f}, "
          f"min {min(nouls):.3f}, max {max(nouls):.3f}, "
          f"stdev {statistics.pstdev(nouls):.3f}")
    print(f"  said Yes (>0.5) for {sum(n > 0.5 for n in nouls)}/{len(nouls)} nodes")
    print(f"importance: mean {statistics.mean(scores):.2f}, "
          f"min {min(scores):.2f}, max {max(scores):.2f}, "
          f"stdev {statistics.pstdev(scores):.2f}")
    print("role picks:")
    for r in sorted(set(roles), key=lambda x: -roles.count(x)):
        print(f"  {r:12s} {roles.count(r):4d}  ({roles.count(r) / len(roles) * 100:.0f}%)")


if __name__ == "__main__":
    sys.exit(main())
