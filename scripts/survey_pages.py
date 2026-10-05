"""
Survey which sites SurfMate actually works well on.

"Works" is not "it returned something". With the 270M the region type carries
no information, so what decides whether the overlay is useful is almost
entirely how well the page describes itself: landmark elements, ARIA roles,
aria-labels and headings. This scores exactly that, per page:

  containers   how many numbered destinations came back (want ~5-9)
  grounded     labels lifted from the page (aria-label / heading) rather than
               a generic "Section"/"Navigation - <text dump>" fallback
  dom-typed    regions whose type the markup stated, so it cannot be wrong
  dropped      containers whose selector failed to resolve, silently discarded

    python3 scripts/surfmate_server.py     # in one shell
    python3 scripts/survey_pages.py        # in another
"""

import argparse
import json
import os
import sys
import time
import urllib.request

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from surfmate_server import ROLE_LABELS, dom_role  # noqa: E402

SURFMATE = os.path.expanduser("~/Developers/SurfMate")
OUT = os.path.join(ROOT, "data", "surfmate", "survey.json")

# (category, name, url). Categories exist so patterns show up: what predicts a
# good result here is how well a page describes itself, and that correlates far
# more with the kind of site than with the technology it is built on.
SITES = [
    # --- reference / documentation -----------------------------------------
    ("docs", "wikipedia", "https://en.wikipedia.org/wiki/Web_accessibility"),
    ("docs", "wikipedia-ko", "https://ko.wikipedia.org/wiki/웹_접근성"),
    ("docs", "mdn", "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API"),
    ("docs", "django-docs", "https://docs.djangoproject.com/en/stable/"),
    ("docs", "chrome-docs", "https://developer.chrome.com/docs/extensions"),
    ("docs", "python-docs", "https://docs.python.org/3/"),
    ("docs", "rust-book", "https://doc.rust-lang.org/book/"),
    ("docs", "react-dev", "https://react.dev/learn"),
    ("docs", "tailwind", "https://tailwindcss.com/docs/installation"),
    ("docs", "postgres", "https://www.postgresql.org/docs/current/index.html"),
    ("docs", "jax-docs", "https://docs.jax.dev/en/latest/"),
    ("docs", "w3c-tr", "https://www.w3.org/TR/"),
    ("docs", "whatwg", "https://html.spec.whatwg.org/multipage/"),
    ("docs", "kernel", "https://kernel.org/"),
    # --- government / institutional ----------------------------------------
    ("gov", "gov-uk", "https://www.gov.uk/"),
    ("gov", "canada", "https://www.canada.ca/en.html"),
    ("gov", "usa-gov", "https://www.usa.gov/"),
    ("gov", "nasa", "https://www.nasa.gov/"),
    ("gov", "weather-gov", "https://www.weather.gov/"),
    ("gov", "europa", "https://european-union.europa.eu/index_en"),
    ("gov", "nih", "https://www.nih.gov/"),
    ("gov", "who", "https://www.who.int/"),
    ("gov", "korea-gov", "https://www.korea.kr/"),
    ("gov", "seoul", "https://www.seoul.go.kr/"),
    ("gov", "ietf", "https://www.ietf.org/"),
    # --- universities -------------------------------------------------------
    ("edu", "mit", "https://www.mit.edu/"),
    ("edu", "stanford", "https://www.stanford.edu/"),
    ("edu", "ox-ac-uk", "https://www.ox.ac.uk/"),
    ("edu", "kaist", "https://www.kaist.ac.kr/kr/"),
    ("edu", "mit-ocw", "https://ocw.mit.edu/"),
    # --- news ---------------------------------------------------------------
    ("news", "bbc", "https://www.bbc.com/news"),
    ("news", "apnews", "https://apnews.com/"),
    ("news", "npr", "https://www.npr.org/"),
    ("news", "guardian", "https://www.theguardian.com/international"),
    ("news", "aljazeera", "https://www.aljazeera.com/"),
    ("news", "dw", "https://www.dw.com/en/top-stories/s-9097"),
    ("news", "yonhap", "https://en.yna.co.kr/"),
    ("news", "hani", "https://www.hani.co.kr/"),
    ("news", "techcrunch", "https://techcrunch.com/"),
    ("news", "arstechnica", "https://arstechnica.com/"),
    # --- developer platforms ------------------------------------------------
    ("devplat", "github", "https://github.com/explore"),
    ("devplat", "gitlab", "https://gitlab.com/explore"),
    ("devplat", "npm", "https://www.npmjs.com/package/react"),
    ("devplat", "huggingface", "https://huggingface.co/models"),
    ("devplat", "arxiv", "https://arxiv.org/list/cs.CL/recent"),
    ("devplat", "stackexchange", "https://stackexchange.com/"),
    ("devplat", "codeberg", "https://codeberg.org/explore/repos"),
    ("devplat", "sourceforge", "https://sourceforge.net/"),
    ("devplat", "readthedocs", "https://readthedocs.org/"),
    ("devplat", "pypi-home", "https://pypi.org/"),
    # --- saas / marketing landing pages -------------------------------------
    ("saas", "stripe", "https://stripe.com/"),
    ("saas", "vercel", "https://vercel.com/"),
    ("saas", "cloudflare", "https://www.cloudflare.com/"),
    ("saas", "notion", "https://www.notion.com/"),
    ("saas", "figma", "https://www.figma.com/"),
    ("saas", "linear", "https://linear.app/"),
    ("saas", "anthropic", "https://www.anthropic.com/"),
    ("saas", "openai", "https://openai.com/"),
    # --- commerce / listings ------------------------------------------------
    ("shop", "books-scrape", "https://books.toscrape.com/"),
    ("shop", "openlibrary", "https://openlibrary.org/"),
    ("shop", "gutenberg", "https://www.gutenberg.org/"),
    ("shop", "archive", "https://archive.org/"),
    ("shop", "yc-companies", "https://www.ycombinator.com/companies"),
    ("shop", "craigslist", "https://sfbay.craigslist.org/"),
    ("shop", "webscraper-shop", "https://webscraper.io/test-sites/e-commerce/allinone"),
    ("shop", "ikea", "https://www.ikea.com/us/en/"),
    ("shop", "etsy", "https://www.etsy.com/"),
    ("shop", "bandcamp", "https://bandcamp.com/"),
    # --- community / forums -------------------------------------------------
    ("forum", "hackernews", "https://news.ycombinator.com/"),
    ("forum", "lobsters", "https://lobste.rs/"),
    ("forum", "discourse", "https://meta.discourse.org/"),
    ("forum", "slashdot", "https://slashdot.org/"),
    ("forum", "clien", "https://www.clien.net/service/"),
    # --- korean portals / services ------------------------------------------
    ("korea", "naver", "https://www.naver.com/"),
    ("korea", "daum", "https://www.daum.net/"),
    ("korea", "kakao", "https://www.kakaocorp.com/page/"),
    ("korea", "coupang-home", "https://www.coupang.com/"),
    ("korea", "melon", "https://www.melon.com/"),
    ("korea", "namu", "https://namu.wiki/"),
    # --- misc / legacy ------------------------------------------------------
    ("misc", "python", "https://www.python.org/"),
    ("misc", "rust-blog", "https://blog.rust-lang.org/"),
    ("misc", "acm", "https://www.acm.org/"),
    ("misc", "ieee", "https://www.ieee.org/"),
    ("misc", "unicode", "https://home.unicode.org/"),
    ("misc", "apache", "https://www.apache.org/"),
    ("misc", "gnu", "https://www.gnu.org/"),
    ("misc", "motherfucking", "https://motherfuckingwebsite.com/"),
]

CHROME_SHIM = """
window.chrome = {
  runtime: { onMessage: { addListener: () => {} }, sendMessage: () => {},
             getURL: (p) => p, lastError: null },
  storage: { local: { get: (k, cb) => cb && cb({}), set: (o, cb) => cb && cb() },
             onChanged: { addListener: () => {} } },
};
"""


def post(url, payload, timeout=300):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def score_page(result, snapshot, lang="en"):
    by_selector = {e.get("selector"): e for e in snapshot.get("elements", [])}
    generic = set(ROLE_LABELS[lang].values())

    containers = result.get("containers", [])
    grounded, dom_typed, dropped = 0, 0, 0
    details = []

    for c in containers:
        el = by_selector.get(c["selector"])
        if el is None:
            dropped += 1
            continue

        label = c["label"]
        # A grounded label came from aria-label or a heading. The fallbacks are
        # either the bare region name or "Region - <first words of the text>".
        is_generic = label in generic or any(
            label.startswith(g + " - ") or label.startswith(g + " ") for g in generic)
        if not is_generic:
            grounded += 1

        if dom_role(el) is not None:
            dom_typed += 1

        details.append({"label": label, "type": c["type"], "grounded": not is_generic})

    n = len(containers)
    return {
        "containers": n,
        "grounded": grounded,
        "grounded_pct": round(grounded / n * 100) if n else 0,
        "dom_typed": dom_typed,
        "dom_typed_pct": round(dom_typed / n * 100) if n else 0,
        "unresolved": dropped,
        "details": details,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright

    with open(os.path.join(SURFMATE, "content.js")) as f:
        content_js = f.read()

    sites = SITES[:args.limit] if args.limit else SITES
    rows = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(
            viewport={"width": 1440, "height": 900},
            user_agent=("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/125.0.0.0 Safari/537.36"))

        for category, name, url in sites:
            page = ctx.new_page()
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
                page.wait_for_timeout(1800)
                page.evaluate(CHROME_SHIM)
                page.evaluate(content_js)
                snap = page.evaluate("generateDOMSnapshot()")
            except Exception as e:
                print(f"{name:16s} LOAD FAILED  {str(e).splitlines()[0][:55]}")
                page.close()
                continue
            page.close()

            t0 = time.time()
            try:
                res = post(f"{args.server}/analyze_page", {
                    "domSnapshot": snap, "url": url,
                    "title": snap.get("title", ""), "language": "en"})
            except Exception as e:
                print(f"{name:16s} SERVER FAILED  {str(e)[:55]}")
                continue
            ms = (time.time() - t0) * 1000

            s = score_page(res, snap)
            s.update(name=name, url=url, category=category, ms=round(ms),
                     title=snap.get("title", "")[:50])
            rows.append(s)
            print(f"{name:16s} containers={s['containers']:2d}  "
                  f"grounded={s['grounded']:2d} ({s['grounded_pct']:3d}%)  "
                  f"dom-typed={s['dom_typed_pct']:3d}%  "
                  f"unresolved={s['unresolved']}  {s['ms']:5d}ms")

        browser.close()

    # A page is worth navigating if it offers a real set of destinations and
    # most of them carry a name taken from the page itself.
    for r in rows:
        r["usable"] = r["containers"] >= 4 and r["grounded_pct"] >= 50

    rows.sort(key=lambda r: (-r["grounded_pct"], -r["containers"]))
    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)

    good = [r for r in rows if r["usable"]]
    print(f"\n{'='*78}")
    print(f"usable: {len(good)}/{len(rows)} pages surveyed "
          f"({len(sites) - len(rows)} failed to load)\n")

    print(f"{'site':17s} {'cat':8s} {'cont':>4} {'grnd':>5} {'dom':>5}  example labels")
    for r in rows:
        mark = "OK " if r["usable"] else "   "
        ex = ", ".join(d["label"][:18] for d in r["details"][:3])
        print(f"{mark}{r['name']:14s} {r['category']:8s} {r['containers']:4d} "
              f"{r['grounded_pct']:4d}% {r['dom_typed_pct']:4d}%  {ex[:52]}")

    print(f"\n{'category':10s} {'pages':>5} {'usable':>7} {'avg grounded':>13} {'avg dom-typed':>14}")
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    for cat, rs in sorted(cats.items(),
                          key=lambda kv: -sum(x["grounded_pct"] for x in kv[1]) / len(kv[1])):
        ok = sum(1 for x in rs if x["usable"])
        print(f"{cat:10s} {len(rs):5d} {ok:7d} "
              f"{sum(x['grounded_pct'] for x in rs) / len(rs):12.0f}% "
              f"{sum(x['dom_typed_pct'] for x in rs) / len(rs):13.0f}%")

    close = [r for r in rows if r["dom_typed_pct"] >= 60 and r["grounded_pct"] < 50]
    empty = [r for r in rows if r["containers"] == 0]
    print(f"\ntyped but unnamed (label fallback would fix): {len(close)}  "
          f"{[r['name'] for r in close]}")
    print(f"nothing found at all: {len(empty)}  {[r['name'] for r in empty]}")
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    raise SystemExit(main())
