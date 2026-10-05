"""
Cruise a week of Hugging Face Daily Papers with one filter criterion.

Each day is a fresh page of ~15 paper titles; the filter scores every one and
keeps the ones matching whatever you are looking for. Nothing in the markup
says which paper is about what, so every verdict here is the model's.

    python3 scripts/surfmate_server.py       # in one shell
    python3 scripts/record_hf_papers.py      # in another
    python3 scripts/record_hf_papers.py --criterion "This is about agents." --days 5
"""

import argparse
import datetime as dt
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--criterion",
                    default="This is about reinforcement learning or training with rewards.")
    ap.add_argument("--end", default=None, help="last date, YYYY-MM-DD (default: today)")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--hold", type=float, default=3.0, help="seconds on each day's result")
    ap.add_argument("--jumps", type=int, default=2, help="n presses per day")
    ap.add_argument("--name", default="hf_papers_week")
    args = ap.parse_args()

    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    dates = [(end - dt.timedelta(days=i)).isoformat()
             for i in range(args.days - 1, -1, -1)]
    print(f"criterion: {args.criterion}")
    print(f"dates: {dates[0]} .. {dates[-1]}\n")

    profile = tempfile.mkdtemp(prefix="surfmate-hf-")
    video_dir = tempfile.mkdtemp(prefix="surfmate-hfvid-")
    summary = []

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
        trim = None

        for i, date in enumerate(dates):
            url = f"https://huggingface.co/papers/date/{date}"
            try:
                page.goto(url, timeout=45000, wait_until="load")
            except Exception:
                try:
                    page.goto(url, timeout=30000, wait_until="domcontentloaded")
                except Exception:
                    print(f"  {date}  navigation failed")
                    continue
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass
            page.wait_for_timeout(1400)
            page.bring_to_front()

            # Hugging Face sends dates with no papers -- weekends, mostly --
            # back to the most recent day that has some. Recording those
            # produces the same page several times over.
            landed = page.evaluate("() => location.pathname")
            if not landed.endswith(date):
                print(f"  {date}  redirected to {landed.split('/')[-1]}, skipping")
                continue

            # The filter bar survives navigation only if it is reopened, since
            # the content script is reloaded with each page.
            sw.evaluate(
                """async () => {
                    const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
                    chrome.tabs.sendMessage(tab.id, {type: 'toggleFilter'}).catch(() => {});
                }"""
            )
            page.wait_for_timeout(700)
            if not page.evaluate("() => !!document.querySelector('.browse-filter-bar')"):
                print(f"  {date}  filter bar did not open")
                continue

            if trim is None:
                # Recording starts here: everything before was setup.
                trim = max(0.0, time.time() - started - 1.2)

            page.fill(".browse-filter-input", args.criterion)
            page.wait_for_timeout(300)
            page.press(".browse-filter-input", "Enter")

            deadline = time.time() + 180
            while time.time() < deadline:
                if page.evaluate("() => document.querySelectorAll('.browse-filter-tick').length"):
                    break
                page.wait_for_timeout(350)

            stats = page.evaluate(
                """() => ({
                    count: document.querySelector('.browse-filter-count')?.textContent || '',
                    kept: [...document.querySelectorAll('.browse-filtered-in')]
                        .map(e => ({
                            s: e.querySelector(':scope > .browse-filter-score')?.textContent || '',
                            t: (e.innerText || '').replace(/\\s+/g, ' ').trim().slice(0, 64),
                        })).slice(0, 6),
                })"""
            )
            print(f"  {date}  {stats['count']}")
            for k in stats["kept"]:
                print(f"       {k['s']}  {k['t']}")
            summary.append((date, stats["count"], stats["kept"]))

            page.wait_for_timeout(int(args.hold * 1000))
            for _ in range(args.jumps):
                page.keyboard.press("n")
                page.wait_for_timeout(1500)

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
        ["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{trim or 0:.2f}", "-i", saved,
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
