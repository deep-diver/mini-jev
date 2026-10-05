"""
Record the live scan as a video.

Playwright records the real page at its own frame rate, so unlike the
frame-by-frame capture scripts this shows the animation at its true speed --
screenshotting is slow enough to distort the timing it is trying to document.

    python3 scripts/surfmate_server.py    # in one shell
    python3 scripts/record_scan.py        # in another
    python3 scripts/record_scan.py --url https://kernel.org/ --pace 200
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


def to_gif(src, dst, fps=16, width=860, start=0.0):
    """Two-pass palette conversion; a single pass makes the glow look banded."""
    palette = os.path.join(tempfile.gettempdir(), "scan_palette.png")
    common = f"fps={fps},scale={width}:-1:flags=lanczos"
    seek = ["-ss", f"{start:.2f}"] if start > 0 else []
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *seek, "-i", src,
             "-vf", f"{common},palettegen=stats_mode=diff", palette],
            check=True)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *seek, "-i", src, "-i", palette,
             "-lavfi", f"{common} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3",
             dst],
            check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="https://en.wikipedia.org/wiki/Web_accessibility")
    ap.add_argument("--name", help="output basename (defaults to the host)")
    ap.add_argument("--server", default="http://127.0.0.1:8777")
    ap.add_argument("--pace", type=int, default=150,
                    help="ms per element of playback pacing (0 = model's real cadence)")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--settle", type=int, default=1200,
                    help="extra ms after load for lazy content to land")
    ap.add_argument("--lead-in", type=float, default=0.7,
                    help="seconds of the settled page to keep before the scan starts")
    ap.add_argument("--tail", type=int, default=4000, help="ms to keep recording after the scan")
    ap.add_argument("--wait", type=float, default=120.0,
                    help="seconds to wait for the scan to finish")
    ap.add_argument("--width", type=int, default=1440)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--no-gif", action="store_true")
    ap.add_argument("--gif-fps", type=int, default=16)
    ap.add_argument("--gif-width", type=int, default=860)
    args = ap.parse_args()

    name = args.name or args.url.split("//")[-1].split("/")[0].replace("www.", "").replace(".", "_")
    os.makedirs(OUT_DIR, exist_ok=True)
    profile = tempfile.mkdtemp(prefix="surfmate-rec-")
    video_dir = tempfile.mkdtemp(prefix="surfmate-vid-")

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            profile,
            headless=False,
            viewport={"width": args.width, "height": args.height},
            record_video_dir=video_dir,
            record_video_size={"width": args.width, "height": args.height},
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
            """([server, lang, pace]) => new Promise(r => chrome.storage.local.set({
                provider: 'local', model: 'google/gemma-3-270m-it',
                localServerUrl: server, language: lang,
                extensionEnabled: true, scanPaceMs: pace,
            }, r))""",
            [args.server, args.lang, args.pace],
        )
        popup.wait_for_timeout(500)
        popup.close()

        page = ctx.new_page()
        # Recording begins when the page is created, so everything up to here
        # is navigation and loading. Note how long that took and cut it off
        # afterwards, keeping only a short lead-in of the settled page.
        record_t0 = time.time()
        try:
            page.goto(args.url, timeout=45000, wait_until="load")
        except Exception:
            # Plenty of real sites never fire `load`: a hung tracker or a video
            # that keeps buffering is enough. The DOM is usually complete well
            # before that, so fall back rather than abandoning the recording.
            print("  load never fired; falling back to domcontentloaded")
            try:
                page.goto(args.url, timeout=30000, wait_until="domcontentloaded")
            except Exception:
                pass  # already navigating; the settle wait below covers it
        try:
            page.wait_for_load_state("networkidle", timeout=15000)
        except Exception:
            pass  # some pages poll forever and never go idle
        page.wait_for_timeout(args.settle)
        page.bring_to_front()
        page.wait_for_timeout(300)
        trim_from = max(0.0, time.time() - record_t0 - args.lead_in)

        sw.evaluate(
            """async () => {
                const [tab] = await chrome.tabs.query({active: true, lastFocusedWindow: true});
                chrome.tabs.sendMessage(tab.id, {type: 'toggleBrowse', enabled: true}).catch(() => {});
            }"""
        )

        # Wait for the numbered containers, then hold on the result. Polled
        # from here rather than with wait_for_function: that helper kept
        # raising immediately instead of waiting, which silently cut every
        # recording short of the thing it was meant to capture.
        deadline = time.time() + args.wait
        badges = 0
        while time.time() < deadline:
            try:
                badges = page.evaluate(
                    "() => document.querySelectorAll('.browse-container-badge').length")
            except Exception as e:
                print(f"  poll failed: {str(e).splitlines()[0][:70]}")
                break
            if badges:
                break
            page.wait_for_timeout(400)
        if not badges:
            print(f"warning: no containers after {args.wait}s; recording anyway")
        else:
            print(f"  {badges} containers after {args.wait - (deadline - time.time()):.1f}s")
        page.wait_for_timeout(args.tail)

        video = page.video
        page.close()
        ctx.close()
        saved = video.path() if video else None

    if not saved:
        candidates = glob.glob(os.path.join(video_dir, "*.webm"))
        saved = candidates[0] if candidates else None

    if not saved or not os.path.exists(saved):
        print("no video produced")
        shutil.rmtree(profile, ignore_errors=True)
        return 1

    webm = os.path.join(OUT_DIR, f"scan_{name}.webm")
    shutil.move(saved, webm)
    size_mb = os.path.getsize(webm) / 1e6
    print(f"video: {webm}  ({size_mb:.1f} MB)")

    mp4 = os.path.join(OUT_DIR, f"scan_{name}.mp4")
    seek = ["-ss", f"{trim_from:.2f}"] if trim_from > 0 else []
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *seek, "-i", webm,
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
             mp4],
            check=True)
        print(f"mp4  : {mp4}  ({os.path.getsize(mp4) / 1e6:.1f} MB, "
              f"trimmed {trim_from:.1f}s of page load)")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("mp4  : skipped (ffmpeg unavailable)")

    if not args.no_gif:
        gif = os.path.join(OUT_DIR, f"scan_{name}.gif")
        if to_gif(webm, gif, fps=args.gif_fps, width=args.gif_width, start=trim_from):
            print(f"gif  : {gif}  ({os.path.getsize(gif) / 1e6:.1f} MB)")

    shutil.rmtree(profile, ignore_errors=True)
    shutil.rmtree(video_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
