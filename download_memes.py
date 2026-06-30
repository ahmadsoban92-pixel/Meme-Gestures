#!/usr/bin/env python3
"""
download_memes.py
-----------------
Downloads one meme image per gesture into a  memes/  folder.

Strategy (tries each in order until one works):
  1. Imgflip public API  — free, no login, ~100 classic templates
  2. Hardcoded direct URLs — a curated backup list of known-good images
  3. Manual instructions  — tells you exactly what to do if both fail

Run once before starting the app:
    python download_memes.py
"""

import json
import pathlib
import sys
import urllib.request
import urllib.error

MEMES_DIR = pathlib.Path("memes")
MEMES_DIR.mkdir(exist_ok=True)

# ── Gesture → Imgflip template name (substring match) ────────────────────────
GESTURE_TO_TEMPLATE = {
    "thumbs_up":  "One Does Not Simply",
    "victory":    "Success Kid",
    "fist":       "Challenge Accepted",
    "ok":         "This Is Fine",
    "pointing":   "Spider-Man Pointing at Spider",
    "namaste":    "Ancient Aliens",
    "smile":      "Laughing Leo",
    "wink":       "I See What You Did There",
    "mouth_open": "Surprised Pikachu",
}

# ── Fallback: direct image URLs (used if the API is blocked) ─────────────────
# These are the actual template image files served by Imgflip.
DIRECT_URLS = {
    "thumbs_up":  "https://i.imgflip.com/1bij.jpg",   # One Does Not Simply
    "victory":    "https://i.imgflip.com/1bhk.jpg",   # Success Kid
    "fist":       "https://i.imgflip.com/1bip.jpg",   # Challenge Accepted
    "ok":         "https://i.imgflip.com/wxica.jpg",  # This Is Fine
    "pointing":   "https://i.imgflip.com/4t0m5.jpg",  # Spider-Man Pointing
    "namaste":    "https://i.imgflip.com/26jxvz.jpg", # Ancient Aliens
    "smile":      "https://i.imgflip.com/1ur9b0.jpg", # Laughing Leo
    "wink":       "https://i.imgflip.com/1e7ql7.jpg", # I See What You Did There
    "mouth_open": "https://i.imgflip.com/2kbn1e.jpg", # Surprised Pikachu
}

# A real browser User-Agent — many APIs reject Python's default one
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def fetch_bytes(url: str) -> bytes:
    """Download a URL with a browser-like User-Agent."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        return resp.read()


def save_url(url: str, dest: pathlib.Path) -> None:
    """Download url and write to dest."""
    data = fetch_bytes(url)
    dest.write_bytes(data)


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Imgflip API
# ─────────────────────────────────────────────────────────────────────────────

def try_imgflip_api() -> "dict | None":
    """Return {gesture: url} from Imgflip API, or None if unreachable."""
    print("📥  Trying Imgflip API…")
    try:
        raw       = fetch_bytes("https://api.imgflip.com/get_memes")
        templates = json.loads(raw)["data"]["memes"]
        print(f"    Got {len(templates)} templates.\n")
    except Exception as exc:
        print(f"    ⚠️  API failed: {exc}\n")
        return None

    result = {}
    for gesture, search in GESTURE_TO_TEMPLATE.items():
        match = next(
            (t for t in templates if search.lower() in t["name"].lower()), None
        )
        if match:
            result[gesture] = match["url"]
        else:
            print(f"    ⚠️  No template matched '{search}' — will try direct URL")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Hardcoded direct URLs
# ─────────────────────────────────────────────────────────────────────────────

def try_direct_urls() -> "dict | None":
    """Return DIRECT_URLS if at least one image is reachable, else None."""
    print("📥  Trying direct image URLs…")
    # Test one URL to see if the host is reachable
    test_url = next(iter(DIRECT_URLS.values()))
    try:
        fetch_bytes(test_url)          # quick reachability check
        print("    Direct URLs reachable.\n")
        return DIRECT_URLS
    except Exception as exc:
        print(f"    ⚠️  Direct URLs also failed: {exc}\n")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Work out which gestures still need images
    needed = {
        g: True for g in GESTURE_TO_TEMPLATE
        if not (MEMES_DIR / f"{g}.jpg").exists()
    }

    if not needed:
        print("✅  All meme images already downloaded — nothing to do.")
        print("   Run  python main.py  to start the app.")
        return

    already = len(GESTURE_TO_TEMPLATE) - len(needed)
    if already:
        print(f"ℹ️   {already} image(s) already present, downloading the rest.\n")

    # Try strategies in order
    url_map = try_imgflip_api() or try_direct_urls()

    if url_map is None:
        _print_manual_instructions()
        return

    # Download
    downloaded = failed = 0
    for gesture in needed:
        dest = MEMES_DIR / f"{gesture}.jpg"
        url  = url_map.get(gesture) or DIRECT_URLS.get(gesture)

        if not url:
            print(f"  ⚠️  No URL for {gesture} — skipping")
            failed += 1
            continue

        try:
            save_url(url, dest)
            print(f"  ✅  {gesture}.jpg")
            downloaded += 1
        except Exception as exc:
            print(f"  ❌  {gesture}: {exc}")
            failed += 1

    # Summary
    print(f"\n{'─'*45}")
    print(f"  Downloaded : {downloaded}")
    print(f"  Failed     : {failed}")
    print(f"{'─'*45}")

    if downloaded:
        print("\n✅  Done!  Run  python main.py  to start the app.")
    else:
        _print_manual_instructions()


def _print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════╗
║  Both download methods failed (likely a network/firewall ║
║  issue).  You can add images MANUALLY in 3 steps:        ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  1. Create a folder called  memes  next to main.py       ║
║                                                          ║
║  2. Download any JPG images you like and name them:      ║
║       memes/thumbs_up.jpg                                ║
║       memes/victory.jpg                                  ║
║       memes/fist.jpg                                     ║
║       memes/ok.jpg                                       ║
║       memes/pointing.jpg                                 ║
║       memes/namaste.jpg                                  ║
║       memes/smile.jpg                                    ║
║       memes/wink.jpg                                     ║
║       memes/mouth_open.jpg                               ║
║                                                          ║
║  3. Run  python main.py  — any gesture with a matching   ║
║     JPG will show the real image; others show the        ║
║     coloured-card fallback.                              ║
║                                                          ║
║  Tip: search Google Images for each meme name, save      ║
║  the image, and rename it to the filename above.         ║
╚══════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
