"""
Benchmark the SurfMate decision task on whatever device JAX finds.

Runs the identical 45-node evaluation set used locally, so the accuracy printed
here must match the local run. If it does not, the TPU numbers are measuring a
different computation and the latency figures mean nothing -- that check is the
point of reporting accuracy in a latency benchmark.

Then sweeps batch size to find where the accelerator stops being fed.

    python3 scripts/tpu_bench_surfmate.py --model google/gemma-4-E2B-it
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

from compare_models import build_set  # noqa: E402
from surfmate_server import PAGE_QUESTIONS, ROLE_ALIASES  # noqa: E402

# Local Mac CPU reference, measured with the same script and the same 45 nodes.
LOCAL_CPU_MS_PER_NODE = {
    "google/gemma-3-270m-it": 112.4,
    "google/gemma-4-E2B-it": 1490.1,
}
LOCAL_ACCURACY = {
    "google/gemma-3-270m-it": 0.267,
    "google/gemma-4-E2B-it": 0.889,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 8, 16, 32, 48])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "surfmate", "tpu_bench.json"))
    args = ap.parse_args()

    import jax
    devices = jax.devices()
    print(f"JAX {jax.__version__}  platform={devices[0].platform}  devices={len(devices)}")
    for d in devices:
        print(f"  {d}")

    items = build_set(args.limit)
    states = [it["state"] for it in items]
    print(f"\n{len(items)} nodes from the shared evaluation set")

    from mini_jev import MiniJevClient

    t0 = time.time()
    client = MiniJevClient(model=args.model)
    load_s = time.time() - t0
    print(f"model ready in {load_s:.1f}s on {client.num_devices} device(s)\n")

    # --- correctness cross-check against the local run -------------------
    responses = client.evaluate_many(states, PAGE_QUESTIONS, chunk_size=len(states))
    preds = [ROLE_ALIASES.get(r.answers["role"].choice, r.answers["role"].choice)
             for r in responses]
    correct = sum(p == it["truth"] for p, it in zip(preds, items))
    accuracy = correct / len(items)
    expected = LOCAL_ACCURACY.get(args.model)
    verdict = "n/a"
    if expected is not None:
        verdict = "MATCHES local" if abs(accuracy - expected) < 0.03 else "DIFFERS from local!"
    print(f"accuracy {accuracy:.1%} ({correct}/{len(items)})  "
          f"local was {expected:.1%}  -> {verdict}\n" if expected
          else f"accuracy {accuracy:.1%}\n")

    # --- latency sweep ---------------------------------------------------
    print(f"{'batch':>6} {'ms/node':>10} {'nodes/s':>9} {'page(19)':>10}")
    sweep = []
    for bs in args.batches:
        if bs > len(states):
            continue
        chunk = states[:bs]
        client.evaluate_many(chunk, PAGE_QUESTIONS, chunk_size=bs)  # warm this shape

        runs = []
        for _ in range(args.repeats):
            t0 = time.time()
            client.evaluate_many(chunk, PAGE_QUESTIONS, chunk_size=bs)
            runs.append((time.time() - t0) / bs * 1000)
        ms = statistics.median(runs)
        sweep.append({"batch": bs, "ms_per_node": round(ms, 2),
                      "nodes_per_s": round(1000 / ms, 1),
                      "page_19_s": round(ms * 19 / 1000, 2)})
        print(f"{bs:>6} {ms:>10.2f} {1000 / ms:>9.1f} {ms * 19 / 1000:>9.2f}s")

    best = min(sweep, key=lambda s: s["ms_per_node"])
    ref = LOCAL_CPU_MS_PER_NODE.get(args.model)
    print(f"\nbest: batch {best['batch']}  {best['ms_per_node']} ms/node  "
          f"page of 19 in {best['page_19_s']}s")
    if ref:
        print(f"vs Mac CPU {ref} ms/node  ->  {ref / best['ms_per_node']:.1f}x")

    report = {
        "model": args.model,
        "platform": devices[0].platform,
        "device_count": len(devices),
        "device": str(devices[0]),
        "load_s": round(load_s, 1),
        "accuracy": accuracy,
        "accuracy_local": expected,
        "accuracy_check": verdict,
        "sweep": sweep,
        "best": best,
        "local_cpu_ms_per_node": ref,
        "speedup_vs_local_cpu": round(ref / best["ms_per_node"], 1) if ref else None,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"-> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
