"""
One page, several criteria, back to back.

The point of a runtime criterion is that the same page answers differently
depending on what you ask for, and that no markup anywhere could have answered
it in advance. This records exactly that: load a page once, filter it by one
topic, then retype and filter by another.

    python3 scripts/surfmate_server.py            # in one shell
    python3 scripts/record_filter_multi.py        # in another
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

DEFAULT_CRITERIA = [
    "This is about space or astronomy.",
    "This is about cars or vehicles.",
    "This is about health, medicine, or the human body.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://arstechnica.com/")
    ap.add_argument("--name", default="filter_multi_arstechnica")
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--criteria", nargs="+", default=DEFAULT_CRITERIA)
    ap.add_argument("--hold", type=float, default=3.5, help="seconds on each result")
    ap.add_argument("--jumps", type=int, default=2, help="n presses per criterion")
    ap.add_argument("--lead-in", type=float, default=1.2)
    args = ap.parse_args()

    profile = tempfile.mkdtemp(prefix="surfmate-multi-")
    video_dir = tempfile.mkdtemp(prefix="surfmate-mvid-")
    results = []

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
            page.goto(args.url, timeout=45000, wait_until="load")
        except Exception:
            try:
                page.goto(args.url, timeout=30000, wait_until="domcontentloaded")
            except Exception:
                print("navigation failed")
                return 1
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
        page.wait_for_timeout(800)
        if not page.evaluate("() => !!document.querySelector('.browse-filter-bar')"):
            print("filter bar did not open")
            return 1

        trim = max(0.0, time.time() - started - args.lead_in)

        for i, criterion in enumerate(args.criteria):
            print(f"=== {criterion}")
            # The page is scored again from scratch; only the criterion changed.
            page.fill(".browse-filter-input", criterion)
            page.wait_for_timeout(600)
            page.press(".browse-filter-input", "Enter")

            # Waiting for ticks to exist is wrong on the second criterion:
            # the previous run's ticks are still on screen, so the wait falls
            # straight through and the stats are read mid-scan. Watch the
            # status line instead, which says "reading" and then "scored".
            deadline = time.time() + 180
            saw_reading = False
            while time.time() < deadline:
                status = page.evaluate(
                    "() => document.querySelector('.browse-filter-status')?.innerText || ''")
                if "reading" in status:
                    saw_reading = True
                elif saw_reading and "scored in" in status:
                    break
                page.wait_for_timeout(300)

            stats = page.evaluate(
                """() => ({
                    count: document.querySelector('.browse-filter-count')?.textContent || '',
                    status: (document.querySelector('.browse-filter-status')?.innerText || '')
                              .replace(/\\s+/g, ' '),
                    kept: [...document.querySelectorAll('.browse-filtered-in')]
                        .map(e => ({
                            s: e.querySelector(':scope > .browse-filter-score')?.textContent || '',
                            t: (e.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 62),
                        }))
                        .sort((a, b) => parseFloat(b.s) - parseFloat(a.s))
                        .slice(0, 6),
                })"""
            )
            print(f"    {stats['count']}   {stats['status'].split('n/p')[0].strip()}")
            for k in stats["kept"]:
                print(f"      {k['s']}  {k['t']}")
            results.append((criterion, stats["count"], stats["kept"]))

            page.wait_for_timeout(int(args.hold * 1000))
            for _ in range(args.jumps):
                page.keyboard.press("n")
                page.wait_for_timeout(1600)
            page.wait_for_timeout(600)

        page.wait_for_timeout(1500)
        video = page.video
        page.close()
        ctx.close()
        saved = video.path() if video else None

    if not saved:
        found = glob.glob(os.path.join(video_dir, "*.webm"))
        saved = found[0] if found else None
    if not saved or not os.path.exists(saved):
        print("no video produced")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"{args.name}.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{trim:.2f}", "-i", saved,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "21",
         "-movflags", "+faststart", out],
        check=True)
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip()
    print(f"\n-> {out}  ({os.path.getsize(out) / 1e6:.1f} MB, {float(dur):.0f}s)")

    shutil.rmtree(profile, ignore_errors=True)
    shutil.rmtree(video_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
