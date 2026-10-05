"""
Open a real browser window running SurfMate on the local model, and leave it up.

Unlike the capture and recording scripts this is not headless and does not
close: it cycles through a few sites, runs the scan on each, and holds the
window open so the animation can be watched as it happens.

    python3 scripts/surfmate_server.py   # in one shell
    python3 scripts/live_demo.py         # in another; Ctrl-C to close
"""

import argparse
import os
import shutil
import tempfile
import time

from playwright.sync_api import sync_playwright

SURFMATE = os.path.expanduser("~/Developers/SurfMate")

SITES = [
    ("https://en.wikipedia.org/wiki/Web_accessibility", "en"),
    ("https://kernel.org/", "en"),
    ("https://www.stanford.edu/", "en"),
    ("https://stackexchange.com/", "en"),
    ("https://ko.wikipedia.org/wiki/웹_접근성", "ko"),
]


def main():
    ap = argparse.ArgumentParser()
    # Default to the E2B server: the 270M answers 0.99 to everything, so a
    # live demo driven by it shows the animation working and the decision not.
    ap.add_argument("--server", default="http://127.0.0.1:8781")
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--pace", type=int, default=0, help="0 = auto, else ms per element")
    ap.add_argument("--hold", type=float, default=7.0, help="seconds to hold on each result")
    ap.add_argument("--loops", type=int, default=3, help="times to cycle the site list")
    ap.add_argument("--sites", nargs="+", help="override the site list")
    args = ap.parse_args()

    sites = [(u, "en") for u in args.sites] if args.sites else SITES
    profile = tempfile.mkdtemp(prefix="surfmate-live-")

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
            """([server, model, pace]) => new Promise(r => chrome.storage.local.set({
                provider: 'local', model,
                localServerUrl: server, language: 'en',
                extensionEnabled: true, scanPaceMs: pace,
            }, r))""",
            [args.server, args.model, args.pace],
        )
        popup.wait_for_timeout(500)
        popup.close()

        page = ctx.new_page()
        print(f"live window open -- {len(sites)} sites x {args.loops} loops. Ctrl-C to stop.\n")

        try:
            for loop in range(args.loops):
                for url, lang in sites:
                    print(f"[{loop + 1}/{args.loops}] {url}")
                    try:
                        page.goto(url, timeout=45000, wait_until="load")
                    except Exception:
                        try:
                            page.goto(url, timeout=30000, wait_until="domcontentloaded")
                        except Exception:
                            print("    navigation failed, skipping")
                            continue
                    try:
                        page.wait_for_load_state("networkidle", timeout=12000)
                    except Exception:
                        pass
                    page.wait_for_timeout(1200)
                    page.bring_to_front()

                    sw.evaluate(
                        """async ([lang]) => {
                            await chrome.storage.local.set({ language: lang });
                            const [tab] = await chrome.tabs.query({
                                active: true, lastFocusedWindow: true });
                            chrome.tabs.sendMessage(tab.id, {
                                type: 'toggleBrowse', enabled: true }).catch(() => {});
                        }""",
                        [lang],
                    )

                    deadline = time.time() + 180
                    while time.time() < deadline:
                        try:
                            if page.evaluate(
                                    "() => document.querySelectorAll("
                                    "'.browse-container-badge').length"):
                                break
                        except Exception:
                            break
                        page.wait_for_timeout(400)

                    page.wait_for_timeout(int(args.hold * 1000))
        except KeyboardInterrupt:
            print("\nstopping")
        finally:
            ctx.close()

    shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
