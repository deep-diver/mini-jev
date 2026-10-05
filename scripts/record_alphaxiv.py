"""
Cruise alphaXiv with one criterion, scoring paper titles only.

alphaXiv is a virtualised infinite feed: there is no page of results to score,
so the filter has to go and get them. Each round scrolls to the bottom, waits
for the next batch to load, returns to where it left off, and reads down --
drawing a box on each title as the model scores it, which is what pulls the
page along.

    python3 scripts/surfmate_server.py        # in one shell
    python3 scripts/record_alphaxiv.py        # in another
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
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "data", "video")

# alphaXiv puts a sign-up dialog over the middle of the feed a few seconds in.
# Escape usually closes it; if it is still up when you run this, close it by
# hand before the cruise starts -- it covers the band the scan reads through.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--criterion",
                    default="This is about reinforcement learning, agents, "
                            "or training with rewards.")
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--name", default="alphaxiv_cruise")
    ap.add_argument("--jumps", type=int, default=3, help="n presses at the end")
    args = ap.parse_args()

    profile = tempfile.mkdtemp(prefix="surfmate-ax-")
    video_dir = tempfile.mkdtemp(prefix="surfmate-axvid-")
    shots = os.path.join(ROOT, "data", "shots")
    os.makedirs(shots, exist_ok=True)

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
            page.goto("https://www.alphaxiv.org/", timeout=45000, wait_until="load")
        except Exception:
            page.goto("https://www.alphaxiv.org/", timeout=30000,
                      wait_until="domcontentloaded")
        page.wait_for_timeout(5000)
        page.keyboard.press("Escape")
        page.wait_for_timeout(1200)
        page.bring_to_front()

        trim = max(0.0, time.time() - started - 1.2)

        sw.evaluate(
            """async ({criterion, threshold, rounds}) => {
                const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
                chrome.tabs.sendMessage(tab.id, {
                    type: 'cruiseFeed', criterion,
                    options: {threshold, rounds, itemSelector: 'a[href^="/abs/"]'},
                }).catch(() => {});
            }""",
            {"criterion": args.criterion, "threshold": args.threshold,
             "rounds": args.rounds},
        )

        status = ""
        deadline = time.time() + 600
        shot = 0
        while time.time() < deadline:
            page.wait_for_timeout(1200)
            status = page.evaluate(
                "() => document.querySelector('.browse-cruise-stat')?.textContent || ''")
            if shot < 3 and "read" in status:
                shot += 1
                page.screenshot(path=os.path.join(shots, f"alphaxiv_{shot}.png"))
            if "done" in status:
                break
        print(f"  {status}")

        page.wait_for_timeout(2500)
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
    print(f"-> {out}  ({os.path.getsize(out) / 1e6:.1f} MB, {float(dur):.0f}s)")

    shutil.rmtree(profile, ignore_errors=True)
    shutil.rmtree(video_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
