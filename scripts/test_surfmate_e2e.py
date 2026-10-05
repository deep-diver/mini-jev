"""
End-to-end check of the local SurfMate path.

Loads real pages in Chromium, runs the extension's own generateDOMSnapshot()
against them, posts the result to the running mini-jev server, and prints the
containers that come back. This exercises the real snapshot format rather than
a hand-written approximation, so a mismatch between the extension and the
server shows up here rather than in the browser.

    python3 scripts/surfmate_server.py      # in one shell
    python3 scripts/test_surfmate_e2e.py    # in another
"""

import argparse
import json
import os
import time
import urllib.request

from playwright.sync_api import sync_playwright

SURFMATE = os.path.expanduser("~/Developers/SurfMate")

URLS = [
    "https://en.wikipedia.org/wiki/Web_accessibility",
    "https://news.ycombinator.com/",
    "https://www.python.org/",
    "https://github.com/explore",
    "https://books.toscrape.com/",
]

# content.js is written for a live extension, so give it just enough of the
# chrome.* surface to load and reach generateDOMSnapshot without throwing.
CHROME_SHIM = """
window.chrome = {
  runtime: {
    onMessage: { addListener: () => {} },
    sendMessage: () => {},
    getURL: (p) => p,
    lastError: null,
  },
  storage: {
    local: {
      get: (keys, cb) => cb && cb({}),
      set: (obj, cb) => cb && cb(),
    },
    onChanged: { addListener: () => {} },
  },
};
"""


def post(url, payload):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()

    with open(os.path.join(SURFMATE, "content.js")) as f:
        content_js = f.read()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})

        for url in URLS:
            page = ctx.new_page()
            try:
                page.goto(url, timeout=25000, wait_until="domcontentloaded")
                page.wait_for_timeout(1500)
                page.evaluate(CHROME_SHIM)
                page.evaluate(content_js)
                snapshot = page.evaluate("generateDOMSnapshot()")
            except Exception as e:
                print(f"\n=== {url}\n    snapshot failed: {str(e).splitlines()[0][:120]}")
                page.close()
                continue
            page.close()

            containers_in = [e for e in snapshot["elements"] if e.get("isContainer")]
            print(f"\n=== {url}")
            print(f"    snapshot: {len(snapshot['elements'])} elements, "
                  f"{len(containers_in)} container candidates")

            t0 = time.time()
            try:
                result = post(f"{args.server}/analyze_page", {
                    "domSnapshot": snapshot,
                    "url": url,
                    "title": snapshot.get("title", ""),
                    "language": args.lang,
                })
            except Exception as e:
                print(f"    server error: {e}")
                continue
            dt = time.time() - t0

            print(f"    -> {len(result['containers'])} containers in {dt * 1000:.0f} ms")
            for i, c in enumerate(result["containers"], 1):
                print(f"       {i}. [{c['type']:11s}] {c['label'][:34]:34s} {c['selector'][:52]}")

        browser.close()


if __name__ == "__main__":
    raise SystemExit(main())
