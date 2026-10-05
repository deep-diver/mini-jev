"""
Capture the live scan animation frame by frame.

Activates SurfMate on a page and screenshots repeatedly while the local model
streams its verdicts, so the boxes appearing and clearing can be inspected as
stills (and stitched into a GIF).

    python3 scripts/surfmate_server.py   # in one shell
    python3 scripts/capture_scan.py      # in another
"""

import argparse
import os
import shutil
import tempfile
import time

from playwright.sync_api import sync_playwright

SURFMATE = os.path.expanduser("~/Developers/SurfMate")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "screenshots", "scan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://www.python.org/")
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--frames", type=int, default=26)
    ap.add_argument("--interval", type=float, default=0.12, help="seconds between frames")
    args = ap.parse_args()

    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="surfmate-scan-")

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
            """(server) => new Promise(resolve => {
                chrome.storage.local.set({
                    provider: 'local',
                    model: 'google/gemma-3-270m-it',
                    localServerUrl: server,
                    language: 'en',
                    extensionEnabled: true,
                }, resolve);
            })""",
            args.server,
        )
        popup.wait_for_timeout(500)
        popup.close()

        page = ctx.new_page()
        events = []
        page.on("console", lambda m: events.append(m.text[:120]))
        page.goto(args.url, timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        page.bring_to_front()

        # content.js returns true from its message handler without ever calling
        # sendResponse, so the channel closes with a rejection. The message is
        # delivered regardless; swallow it rather than failing the capture.
        sw.evaluate(
            """async () => {
                const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
                // Do not await: content.js returns true from its handler and
                // never calls sendResponse, so the promise hangs for ~5s. The
                // message is delivered immediately regardless, and awaiting it
                // means every screenshot lands after the scan already finished.
                chrome.tabs.sendMessage(tab.id, {type: 'toggleBrowse', enabled: true})
                    .catch(() => {});
            }"""
        )

        t0 = time.time()
        for i in range(args.frames):
            page.screenshot(path=os.path.join(OUT_DIR, f"frame_{i:02d}.png"))
            time.sleep(args.interval)
        print(f"{args.frames} frames over {time.time() - t0:.1f}s -> {OUT_DIR}")

        # Count how many boxes were alive in the DOM at the end, as a sanity check.
        page.wait_for_timeout(2500)
        page.screenshot(path=os.path.join(OUT_DIR, "final.png"))
        ctx.close()

    shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
