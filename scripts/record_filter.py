"""
Record the relevance filter running on a real page.

Loads the page, opens the filter bar, types a criterion, and follows the scan
down the page while every text block is scored -- then walks the matches with
n/p. Page load is trimmed off the front so the recording opens on the settled
site.

    python3 scripts/surfmate_server.py        # in one shell
    python3 scripts/record_filter.py          # in another; all five sites
    python3 scripts/record_filter.py --only hn
"""

import argparse
import glob
import os
import shutil
import subprocess
import tempfile
import time

from playwright.sync_api import sync_playwright

SURFMATE = os.path.expanduser("~/Developers/SurfMate")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "video")

# A criterion that is genuinely selective on each site: no markup anywhere says
# which of these blocks is about the topic, which is the whole point.
SITES = [
    ("arstechnica", "https://arstechnica.com/",
     "This is about space or astronomy."),
    ("hackernews", "https://news.ycombinator.com/",
     "This is about AI, machine learning, or language models."),
    ("arxiv", "https://arxiv.org/list/cs.CL/recent",
     "This is about evaluation, benchmarks, or measuring model quality."),
    ("bbc", "https://www.bbc.com/news",
     "This is about health, medicine, or disease."),
    ("guardian", "https://www.theguardian.com/international",
     "This is about climate, energy, or the environment."),
]


def record(name, url, criterion, args):
    profile = tempfile.mkdtemp(prefix="surfmate-filter-")
    video_dir = tempfile.mkdtemp(prefix="surfmate-fvid-")

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            profile,
            headless=False,
            viewport={"width": 1440, "height": 900},
            record_video_dir=video_dir,
            record_video_size={"width": 1440, "height": 900},
            args=[
                f"--disable-extensions-except={SURFMATE}",
                f"--load-extension={SURFMATE}",
                "--hide-scrollbars",
            ],
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
        popup.wait_for_timeout(400)
        popup.close()

        page = ctx.new_page()
        started = time.time()
        try:
            page.goto(url, timeout=45000, wait_until="load")
        except Exception:
            print("    load never fired; falling back to domcontentloaded")
            try:
                page.goto(url, timeout=30000, wait_until="domcontentloaded")
            except Exception:
                pass
        try:
            page.wait_for_load_state("networkidle", timeout=12000)
        except Exception:
            pass
        page.wait_for_timeout(1500)
        page.bring_to_front()

        sw.evaluate(
            """async () => {
                const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
                chrome.tabs.sendMessage(tab.id, {type: 'toggleFilter'}).catch(() => {});
            }"""
        )
        page.wait_for_timeout(900)
        if not page.evaluate("() => !!document.querySelector('.browse-filter-bar')"):
            print("    filter bar never appeared")
            ctx.close()
            shutil.rmtree(profile, ignore_errors=True)
            return None

        # Everything before this is setup; the recording starts here.
        trim = time.time() - started - args.lead_in

        page.fill(".browse-filter-input", criterion)
        page.wait_for_timeout(500)
        page.press(".browse-filter-input", "Enter")

        deadline = time.time() + args.wait
        while time.time() < deadline:
            if page.evaluate("() => document.querySelectorAll('.browse-filter-tick').length"):
                break
            page.wait_for_timeout(400)
        scanned = time.time() - started - trim - args.lead_in

        stats = page.evaluate(
            """() => ({
                kept: document.querySelectorAll('.browse-filter-tick').length,
                total: document.querySelectorAll('.browse-scan-box').length,
                count: document.querySelector('.browse-filter-count')?.textContent || '',
                status: (document.querySelector('.browse-filter-status')?.innerText || '')
                          .replace(/\\s+/g, ' '),
            })"""
        )
        page.wait_for_timeout(2200)
        for _ in range(args.jumps):
            page.keyboard.press("n")
            page.wait_for_timeout(1700)
        page.wait_for_timeout(1500)

        video = page.video
        page.close()
        ctx.close()
        saved = video.path() if video else None

    if not saved:
        found = glob.glob(os.path.join(video_dir, "*.webm"))
        saved = found[0] if found else None
    if not saved or not os.path.exists(saved):
        shutil.rmtree(profile, ignore_errors=True)
        return None

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"filter_{name}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{max(trim, 0):.2f}", "-i", saved,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
         "-movflags", "+faststart", out],
        check=True)

    shutil.rmtree(profile, ignore_errors=True)
    shutil.rmtree(video_dir, ignore_errors=True)
    return {"out": out, "scanned": scanned, **stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--only", help="record a single site by name")
    ap.add_argument("--lead-in", type=float, default=1.2,
                    help="seconds of settled page kept before the filter is used")
    ap.add_argument("--wait", type=float, default=180.0)
    ap.add_argument("--jumps", type=int, default=3, help="n presses after the scan")
    args = ap.parse_args()

    sites = [s for s in SITES if not args.only or s[0] == args.only]
    results = []
    for name, url, criterion in sites:
        print(f"=== {name}  \"{criterion}\"")
        try:
            r = record(name, url, criterion, args)
        except Exception as e:
            print(f"    failed: {str(e).splitlines()[0][:90]}")
            continue
        if not r:
            continue
        size = os.path.getsize(r["out"]) / 1e6
        dur = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", r["out"]],
            capture_output=True, text=True).stdout.strip()
        print(f"    {r['count']}   scan {r['scanned']:.0f}s")
        print(f"    -> {r['out']}  ({size:.1f} MB, {float(dur):.0f}s)")
        results.append((name, r["count"], size, float(dur)))

    if results:
        print(f"\n{len(results)} recordings in {OUT_DIR}")
        for name, count, size, dur in results:
            print(f"  filter_{name}.mp4   {count:24s} {size:5.1f} MB  {dur:4.0f}s")


if __name__ == "__main__":
    raise SystemExit(main())
