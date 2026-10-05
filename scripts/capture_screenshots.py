"""
Capture real screenshots of SurfMate running on the local mini-jev server.

Loads the actual unpacked extension into Chromium, points it at the local
provider, activates the overlay on real pages, and saves PNGs. This is the only
check that exercises the overlay itself rather than the JSON contract.

    python3 scripts/surfmate_server.py       # in one shell
    python3 scripts/capture_screenshots.py   # in another
"""

import argparse
import os
import shutil
import tempfile

from playwright.sync_api import sync_playwright

SURFMATE = os.path.expanduser("~/Developers/SurfMate")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "screenshots")

PAGES = [
    ("wikipedia", "https://en.wikipedia.org/wiki/Web_accessibility"),
    ("python_org", "https://www.python.org/"),
    ("hackernews", "https://news.ycombinator.com/"),
    ("books", "https://books.toscrape.com/"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--enter-container", type=int, default=1,
                    help="which numbered container to open for the second shot")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="surfmate-profile-")

    with sync_playwright() as p:
        # Extensions only load in a persistent context, and only in headed mode.
        ctx = p.chromium.launch_persistent_context(
            profile,
            headless=False,
            viewport={"width": 1440, "height": 900},
            args=[
                f"--disable-extensions-except={SURFMATE}",
                f"--load-extension={SURFMATE}",
            ],
        )

        # The service worker starts lazily; wait for it so we can read its id.
        sw = ctx.service_workers[0] if ctx.service_workers else ctx.wait_for_event("serviceworker")
        ext_id = sw.url.split("/")[2]
        print(f"extension id: {ext_id}")

        # Configure through the popup, which runs in the extension's origin and
        # so is allowed to write chrome.storage.
        popup = ctx.new_page()
        popup.goto(f"chrome-extension://{ext_id}/popup.html")
        popup.evaluate(
            """([server, lang]) => new Promise(resolve => {
                chrome.storage.local.set({
                    provider: 'local',
                    model: 'google/gemma-3-270m-it',
                    localServerUrl: server,
                    language: lang,
                    extensionEnabled: true,
                }, resolve);
            })""",
            [args.server, args.lang],
        )
        popup.wait_for_timeout(600)
        popup.screenshot(path=os.path.join(OUT_DIR, "00_popup.png"))
        print("saved 00_popup.png")
        popup.close()

        for name, url in PAGES:
            page = ctx.new_page()
            logs = []
            page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"[:160]))
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(2000)
                page.bring_to_front()

                # The content script lives in an isolated world, so chrome.* is
                # not reachable from page.evaluate. Drive the toggle from the
                # service worker, which can address the tab directly.
                #
                # Match on the active tab rather than on URL: without the "tabs"
                # permission chrome.tabs.query omits url entirely, so a URL match
                # silently finds nothing. Tab ids are always returned.
                result = sw.evaluate(
                    """async () => {
                        const [tab] = await chrome.tabs.query({
                            active: true, lastFocusedWindow: true
                        });
                        if (!tab) return 'no active tab';
                        // Do not await: content.js returns true without ever
                        // calling sendResponse, so this promise hangs ~5s and
                        // every screenshot would land after the scan ended.
                        chrome.tabs.sendMessage(tab.id, {
                            type: 'toggleBrowse', enabled: true
                        }).catch(() => {});
                        return 'sent to tab ' + tab.id;
                    }"""
                )
                print(f"  {name}: {result}")
                # Page analysis runs on the local model; give it room.
                page.wait_for_timeout(12000)

                shot = os.path.join(OUT_DIR, f"{name}_1_containers.png")
                page.screenshot(path=shot)
                print(f"saved {os.path.basename(shot)}")

                page.keyboard.press(str(args.enter_container))
                page.wait_for_timeout(9000)
                shot = os.path.join(OUT_DIR, f"{name}_2_elements.png")
                page.screenshot(path=shot)
                print(f"saved {os.path.basename(shot)}")
            except Exception as e:
                print(f"{name}: {str(e).splitlines()[0][:140]}")
            finally:
                for line in [l for l in logs if "SurfMate" in l or "error" in l.lower()][:12]:
                    print(f"    | {line}")
                page.close()

        ctx.close()

    shutil.rmtree(profile, ignore_errors=True)
    print(f"\n-> {OUT_DIR}")


if __name__ == "__main__":
    raise SystemExit(main())
