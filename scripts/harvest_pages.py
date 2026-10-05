"""
Harvest container candidates from real pages for the SurfMate decision task.

Loads each URL in a headless Chromium, runs scripts/extract_dom.js against the
live DOM, and writes one JSON object per page to data/surfmate/pages.jsonl.

    python3 scripts/harvest_pages.py                  # default site list
    python3 scripts/harvest_pages.py --urls urls.txt  # one URL per line
"""

import argparse
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
EXTRACTOR = os.path.join(ROOT, "scripts", "extract_dom.js")
OUT_DIR = os.path.join(ROOT, "data", "surfmate")

# A spread of layout archetypes: docs, news, listings, forums, landing pages,
# search results, government. Container structure differs sharply across these,
# which is the variation the model has to survive.
DEFAULT_URLS = [
    "https://en.wikipedia.org/wiki/Web_accessibility",
    "https://news.ycombinator.com/",
    "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API",
    "https://www.python.org/",
    "https://arxiv.org/list/cs.CL/recent",
    "https://github.com/explore",
    "https://www.npmjs.com/package/react",
    "https://docs.djangoproject.com/en/stable/",
    "https://apnews.com/",
    "https://www.bbc.com/news",
    "https://www.weather.gov/",
    "https://huggingface.co/models",
    "https://www.gutenberg.org/",
    "https://kernel.org/",
    "https://www.gov.uk/",
    "https://archive.org/",
    "https://www.nasa.gov/",
    # Shop / listing layouts, which have a container structure of their own.
    "https://books.toscrape.com/",
    "https://webscraper.io/test-sites/e-commerce/allinone",
    "https://openlibrary.org/",
    "https://www.ycombinator.com/companies",
    # Docs / blog / spec archetypes.
    "https://blog.rust-lang.org/",
    "https://developer.chrome.com/docs/extensions",
    "https://www.ietf.org/",
    "https://simple.wikipedia.org/wiki/Main_Page",
]
# Dropped: w3.org/WAI, stackoverflow.com, pypi.org -- these serve a Cloudflare
# challenge or CAPTCHA to headless Chromium, so the harvest captures the
# interstitial instead of the page.


def load_urls(path):
    if not path:
        return DEFAULT_URLS
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", help="file with one URL per line")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "pages.jsonl"))
    ap.add_argument("--timeout", type=int, default=25000)
    ap.add_argument("--settle-ms", type=int, default=1500,
                    help="extra wait after load so lazy content lands")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    urls = load_urls(args.urls)
    with open(EXTRACTOR) as f:
        extractor_js = f.read()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ok = 0
    failed = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": args.width, "height": args.height},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
            ),
        )
        with open(args.out, "w") as out:
            for i, url in enumerate(urls, 1):
                page = ctx.new_page()
                try:
                    t0 = time.time()
                    page.goto(url, timeout=args.timeout, wait_until="domcontentloaded")
                    page.wait_for_timeout(args.settle_ms)
                    data = page.evaluate(extractor_js)
                    data["harvested_at"] = time.time()
                    out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    out.flush()
                    ok += 1
                    print(
                        f"[{i}/{len(urls)}] {len(data['nodes']):3d} nodes "
                        f"({data['n_candidates_total']:3d} raw)  "
                        f"{time.time() - t0:.1f}s  {url}"
                    )
                except Exception as e:
                    failed.append((url, str(e).split("\n")[0][:90]))
                    print(f"[{i}/{len(urls)}] FAILED  {url}\n    {failed[-1][1]}")
                finally:
                    page.close()
        browser.close()

    print(f"\n{ok}/{len(urls)} pages -> {args.out}")
    if failed:
        print("failed:")
        for url, err in failed:
            print(f"  {url}  {err}")


if __name__ == "__main__":
    sys.exit(main())
