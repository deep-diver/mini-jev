"""
Measure how much the container labels overlap each other.

Renders the overlay on several pages and reports, per page, how many pairs of
labels intersect and how much area is doubly covered. Also screenshots the
result so the numbers can be eyeballed.

    python3 scripts/surfmate_server.py           # in one shell
    python3 scripts/check_label_overlap.py       # in another
"""

import argparse
import itertools
import json
import os
import shutil
import tempfile

from playwright.sync_api import sync_playwright

SURFMATE = os.path.expanduser("~/Developers/SurfMate")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "screenshots", "labels")

PAGES = [
    ("python_org", "https://www.python.org/"),
    ("wikipedia", "https://en.wikipedia.org/wiki/Web_accessibility"),
    ("github", "https://github.com/explore"),
    ("hackernews", "https://news.ycombinator.com/"),
    ("books", "https://books.toscrape.com/"),
]

# Measure the inner <svg>, not the badge element. The badge is an unsized
# wrapper whose own rect is the full page width at zero height; the circle and
# the label plate are painted by a position:fixed svg inside it.
PROBE = """() => [...document.querySelectorAll('.browse-container-badge > svg')].map(el => {
  const b = el.getBoundingClientRect();
  return {
    label: (el.parentElement.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 40),
    x: Math.round(b.x), y: Math.round(b.y),
    w: Math.round(b.width), h: Math.round(b.height),
  };
})"""


def overlap_area(a, b):
    dx = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    dy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    return dx * dy if dx > 0 and dy > 0 else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--tag", default="after", help="label for this run's output")
    ap.add_argument("--surfmate", default=SURFMATE,
                    help="extension directory to load (used to A/B the placement)")
    args = ap.parse_args()

    out_dir = os.path.join(OUT_DIR, args.tag)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="surfmate-labels-")

    results = []
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            profile, headless=False, viewport={"width": 1440, "height": 900},
            args=[f"--disable-extensions-except={args.surfmate}",
                  f"--load-extension={args.surfmate}"],
        )
        sw = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker")
        ext_id = sw.url.split("/")[2]

        popup = ctx.new_page()
        popup.goto(f"chrome-extension://{ext_id}/popup.html")
        popup.evaluate(
            """(server) => new Promise(r => chrome.storage.local.set({
                provider: 'local', model: 'google/gemma-3-270m-it',
                localServerUrl: server, language: 'en',
                extensionEnabled: true, scanPaceMs: 0,
            }, r))""",
            args.server,
        )
        popup.wait_for_timeout(500)
        popup.close()

        for name, url in PAGES:
            page = ctx.new_page()
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                page.bring_to_front()
                sw.evaluate(
                    """async () => {
                        const [t] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
                        chrome.tabs.sendMessage(t.id, {type: 'toggleBrowse', enabled: true}).catch(() => {});
                    }"""
                )
                page.wait_for_timeout(14000)
                badges = page.evaluate(PROBE)
                page.screenshot(path=os.path.join(out_dir, f"{name}.png"))
            except Exception as e:
                print(f"{name}: {str(e).splitlines()[0][:110]}")
                page.close()
                continue
            page.close()

            pairs = list(itertools.combinations(badges, 2))
            colliding = [(a, b) for a, b in pairs if overlap_area(a, b) > 0]
            total = sum(overlap_area(a, b) for a, b in pairs)
            worst = max((overlap_area(a, b) for a, b in pairs), default=0)

            results.append({
                "page": name, "labels": len(badges),
                "colliding_pairs": len(colliding),
                "total_overlap_px": total, "worst_pair_px": worst,
                "examples": [f"{a['label']} × {b['label']}" for a, b in colliding[:3]],
            })
            print(f"{name:12s} labels={len(badges):2d}  colliding pairs={len(colliding):2d}  "
                  f"overlap={total:6d}px²  worst={worst:5d}px²")
            for ex in results[-1]["examples"]:
                print(f"               {ex}")

        ctx.close()
    shutil.rmtree(profile, ignore_errors=True)

    with open(os.path.join(out_dir, "overlap.json"), "w") as f:
        json.dump(results, f, indent=2)

    tp = sum(r["colliding_pairs"] for r in results)
    ta = sum(r["total_overlap_px"] for r in results)
    print(f"\nTOTAL  colliding pairs={tp}  overlap area={ta}px²   -> {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
