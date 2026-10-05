"""
Verify that the live scan actually corresponds to the final result.

Runs the scan at a deliberately slow pace, screenshots every distinct step, and
records the geometry of every box as it is drawn. At the end it records the
surviving container borders and checks the claim the animation makes:

  1. every element that flashes was really judged (one box per streamed verdict)
  2. every surviving container was one of the elements that flashed
  3. the result pass accounts for every flashed element as survived or cut
  4. the survivors it marks are exactly the containers that get a number

Failing any of these would mean the animation is decorative rather than a view
of the decision, which is the whole point of it.

    python3 scripts/surfmate_server.py    # in one shell
    python3 scripts/verify_scan.py        # in another
"""

import argparse
import json
import os
import shutil
import tempfile
import time

from playwright.sync_api import sync_playwright

SURFMATE = os.path.expanduser("~/Developers/SurfMate")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "screenshots", "verify")

# Read whatever the overlay is showing right now: the transient scan boxes, the
# settled result boxes, and the persistent container borders.
PROBE = """() => {
  // Identify boxes by their data-selector, never by geometry: the flash
  // animation scales them, so a rect sampled twice during one draw differs and
  // would count as two elements.
  return {
    scan: [...document.querySelectorAll('.browse-scan-box:not(.result)')].map(el => ({
      idx: el.dataset.idx,
      selector: el.dataset.selector,
      kept: el.classList.contains('kept'),
      label: el.querySelector('.browse-scan-tag')?.textContent || '',
    })),
    result: [...document.querySelectorAll('.browse-scan-box.result')].map(el => ({
      idx: el.dataset.idx,
      selector: el.dataset.selector,
      survived: el.dataset.survived === '1',
      label: el.querySelector('.browse-scan-tag')?.textContent || '',
    })),
    counter: document.querySelector('.browse-scan-counter')?.innerText.replace(/\\n/g, ' ') || '',
    borders: [...document.querySelectorAll('.browse-container-border[data-selector]')]
              .map(el => el.dataset.selector),
    badges: [...document.querySelectorAll('.browse-container-badge')]
              .map(el => el.textContent.replace(/\\s+/g, ' ').trim()),
  };
}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://www.python.org/")
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--pace", type=int, default=650, help="ms per element, slowed for inspection")
    ap.add_argument("--poll", type=float, default=0.05)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="surfmate-verify-")

    seen_boxes = {}
    result_boxes = {}    # selector -> result-pass box
    steps = []           # one entry per distinct on-screen state
    final = None

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            profile,
            headless=False,
            viewport={"width": 1440, "height": 900},
            args=[
                f"--disable-extensions-except={SURFMATE}",
                f"--load-extension={SURFMATE}",
            ],
        )
        sw = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker")
        ext_id = sw.url.split("/")[2]

        popup = ctx.new_page()
        popup.goto(f"chrome-extension://{ext_id}/popup.html")
        popup.evaluate(
            """([server, pace]) => new Promise(resolve => {
                chrome.storage.local.set({
                    provider: 'local',
                    model: 'google/gemma-3-270m-it',
                    localServerUrl: server,
                    language: 'en',
                    extensionEnabled: true,
                    scanPaceMs: pace,
                }, resolve);
            })""",
            [args.server, args.pace],
        )
        popup.wait_for_timeout(600)
        popup.close()

        page = ctx.new_page()
        page.goto(args.url, timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        page.bring_to_front()

        sw.evaluate(
            """async () => {
                const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
                chrome.tabs.sendMessage(tab.id, {type: 'toggleBrowse', enabled: true}).catch(() => {});
            }"""
        )

        t0 = time.time()
        prev_key = None
        shot = 0
        done_at = None

        while time.time() - t0 < args.seconds:
            state = page.evaluate(PROBE)

            for box in state["scan"]:
                key = box["selector"]
                if key and key not in seen_boxes:
                    seen_boxes[key] = {"kept": box["kept"], "label": box["label"],
                                       "idx": box["idx"]}
            for box in state["result"]:
                if box["selector"]:
                    result_boxes[box["selector"]] = box

            # Screenshot only when what is on screen actually changed, so the
            # frames are steps rather than an arbitrary time sampling.
            key = json.dumps([state["scan"], state["result"], state["counter"],
                              state["borders"]], sort_keys=True)
            if key != prev_key:
                prev_key = key
                path = os.path.join(OUT_DIR, f"step_{shot:03d}.png")
                page.screenshot(path=path)
                steps.append({
                    "t_ms": round((time.time() - t0) * 1000),
                    "file": os.path.basename(path),
                    "visible_boxes": len(state["scan"]),
                    "labels": [b["label"] for b in state["scan"]],
                    "counter": state["counter"],
                    "borders": len(state["borders"]),
                })
                shot += 1

            if state["borders"] and not state["scan"] and not state["result"]:
                if done_at is None:
                    done_at = time.time()
                elif time.time() - done_at > 1.2:
                    final = state
                    break

            time.sleep(args.poll)

        if final is None:
            page.wait_for_timeout(1500)
            final = page.evaluate(PROBE)
        page.screenshot(path=os.path.join(OUT_DIR, "final.png"))
        ctx.close()

    shutil.rmtree(profile, ignore_errors=True)

    # ---------------- verification ----------------
    flashed = set(seen_boxes)
    flashed_passed = {s for s, v in seen_boxes.items() if v["kept"]}
    survived_in_result = {s for s, v in result_boxes.items() if v["survived"]}
    culled_in_result = {s for s, v in result_boxes.items() if not v["survived"]}
    borders = set(final["borders"])

    checks = {
        "every final container was flashed during the scan":
            sorted(borders - flashed) == [],
        "every final container is marked survived in the result pass":
            sorted(borders - survived_in_result) == [],
        "result pass covers exactly the elements that were flashed":
            sorted(flashed ^ set(result_boxes)) == [],
        "survivors in the result pass match the final containers":
            sorted(survived_in_result ^ borders) == [],
        "nothing is both survived and culled":
            sorted(survived_in_result & culled_in_result) == [],
    }

    report = {
        "url": args.url,
        "pace_ms": args.pace,
        "steps_captured": len(steps),
        "flashed": len(flashed),
        "flashed_passed_wrapper_test": len(flashed_passed),
        "result_survived": len(survived_in_result),
        "result_culled": len(culled_in_result),
        "final_containers": len(borders),
        "checks": checks,
        "final_not_flashed": sorted(borders - flashed),
        "final_badges": final["badges"],
        "final_counter": final["counter"],
        "steps": steps,
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nsteps captured   : {len(steps)}  -> {OUT_DIR}")
    print(f"flashed          : {len(flashed)} elements "
          f"({len(flashed_passed)} passed the wrapper test)")
    print(f"result pass      : {len(survived_in_result)} survived, {len(culled_in_result)} cut")
    print(f"final containers : {len(borders)}")
    print(f"final counter    : {final['counter']}")
    print()
    ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    if report["final_not_flashed"]:
        print(f"\n  containers never flashed: {report['final_not_flashed']}")
    print(f"\nbadges: {final['badges']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
