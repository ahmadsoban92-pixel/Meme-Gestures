"""
meme_panel.py — Meme panel renderer  (v2 — real images)
--------------------------------------------------------
If  memes/<gesture>.jpg  exists (put there by download_memes.py), the image
fills the panel and caption text is overlaid in classic meme style.

If no image is found the original coloured-card fallback is shown instead,
so the app always works even without running the download script first.
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np


# ── Caption text + fallback colours ──────────────────────────────────────────
MEMES: dict = {
    "thumbs_up": {
        "top":    "THUMBS",
        "bottom": "UP",
        "bg":     ( 20,  20, 150), "accent": ( 60,  60, 220), "fg": (255, 255, 255),
    },
    "victory": {
        "top":    "ALRIGHT,",
        "bottom": "PEACE OUT",
        "bg":     ( 20, 120,  20), "accent": ( 50, 200,  60), "fg": (255, 255, 255),
    },
    "fist": {
        "top":    "YO BRO,",
        "bottom": "THE RESULTS DROPPED",
        "bg":     ( 15,  15,  15), "accent": ( 50,  50,  50), "fg": (  0, 210, 255),
    },
    "ok": {
        "top":    "THIS IS",
        "bottom": "PERFECTLY FINE",
        "bg":     (160, 100,   0), "accent": (220, 160,  30), "fg": (255, 255, 255),
    },
    "pointing": {
        "top":    "POINTING AT",
        "bottom": "YOU",
        "bg":     (110,  10, 110), "accent": (190,  50, 190), "fg": (255, 255,   0),
    },
    "namaste": {
        "top":    "ACTUALLY I DON'T CODE,",
        "bottom": "I DEVISE SOLUTIONS",
        "bg":     (  5, 110, 110), "accent": ( 30, 190, 190), "fg": (255, 255, 255),
    },
    "smile": {
        "top":    "WHEN YOU FINALLY",
        "bottom": "UNDERSTAND THE CODE",
        "bg":     ( 15,  90,  50), "accent": ( 40, 160, 100), "fg": (255, 255,   0),
    },
    "wink": {
        "top":    "I SEE WHAT",
        "bottom": "YOU DID THERE",
        "bg":     (150,  50,   0), "accent": (230, 110,  30), "fg": (255, 255, 255),
    },
    "mouth_open": {
        "top":    "SURPRISED PIKACHU",
        "bottom": "FACE",
        "bg":     (160, 130,   0), "accent": (230, 200,  30), "fg": (255, 255, 255),
    },
}

DEFAULT_MEME: dict = {
    "top":    "SHOW ME A GESTURE",
    "bottom": "OR EXPRESSION !",
    "bg":     ( 25,  25,  25), "accent": ( 55,  55,  55), "fg": (160, 160, 160),
}

MEMES_DIR = Path("memes")
_FONT     = cv2.FONT_HERSHEY_DUPLEX
_WHITE    = (255, 255, 255)

# Module-level image cache: loaded once, reused every frame
_IMG_CACHE: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_image(gesture: str) -> "Optional[np.ndarray]":
    """Return cached meme image for gesture, or None if the file isn't there."""
    if gesture not in _IMG_CACHE:
        path = MEMES_DIR / f"{gesture}.jpg"
        _IMG_CACHE[gesture] = cv2.imread(str(path)) if path.exists() else None
    return _IMG_CACHE[gesture]


def _place_image(panel: np.ndarray, img: np.ndarray) -> None:
    """
    Scale the image to fill the panel's full width (preserving aspect ratio),
    then centre it vertically.
      • If taller than the panel  → crop to centre.
      • If shorter than the panel → letterbox with black bars.
    """
    ph, pw = panel.shape[:2]
    ih, iw = img.shape[:2]

    scale   = pw / iw
    nw, nh  = pw, int(ih * scale)
    scaled  = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    panel[:] = 0                          # clear to black first
    if nh >= ph:                          # image taller → centre-crop
        y0 = (nh - ph) // 2
        panel[:] = scaled[y0 : y0 + ph, :]
    else:                                 # image shorter → letterbox
        y0 = (ph - nh) // 2
        panel[y0 : y0 + nh, :] = scaled


def _darken_band(panel: np.ndarray, y1: int, y2: int, opacity: float = 0.60) -> None:
    """
    Darken a horizontal strip of the panel so white text stays readable
    on any background image.  opacity=0.60 → strip becomes 40% as bright.
    """
    region = panel[y1:y2].astype(np.float32)
    panel[y1:y2] = np.clip(region * (1.0 - opacity), 0, 255).astype(np.uint8)


def _centered_text(img: np.ndarray, text: str, y: int, scale: float,
                   color: tuple, thickness: int, panel_w: int) -> None:
    """Draw horizontally centred text with a thick black outline."""
    (tw, _), _ = cv2.getTextSize(text, _FONT, scale, thickness)
    x = max(6, (panel_w - tw) // 2)
    cv2.putText(img, text, (x, y), _FONT, scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), _FONT, scale, color,     thickness,     cv2.LINE_AA)


# ── Fallback renderer (no image) ──────────────────────────────────────────────

def _gradient(img, x1, y1, x2, y2, c1, c2):
    n = y2 - y1
    if n <= 0 or x2 <= x1:
        return
    a, b = np.array(c1, dtype=np.float32), np.array(c2, dtype=np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    colors = np.clip(a * (1 - t[:, None]) + b * t[:, None], 0, 255).astype(np.uint8)
    img[y1:y2, x1:x2] = colors[:, None, :]


def _render_fallback(panel: np.ndarray, data: dict,
                     gesture: Optional[str], width: int, height: int) -> None:
    """Original coloured-card design — used when no meme image is found."""
    bg, accent, fg = data["bg"], data["accent"], data["fg"]
    dark = tuple(max(0, c - 50) for c in bg)
    _gradient(panel, 0, 0, width, height, bg, dark)

    ST   = 54                                       # strip height
    pad  = 22
    cl, cr = pad, width - pad
    ct, cb = ST + 12, height - ST - 12

    # Top + bottom strips
    _gradient(panel, 0, 0,          width, ST,     (0, 0, 0), (18, 18, 18))
    _gradient(panel, 0, height - ST, width, height, (18, 18, 18), (0, 0, 0))
    _centered_text(panel, data["top"],    37,          0.76, fg, 2, width)
    _centered_text(panel, data["bottom"], height - 16, 0.76, fg, 2, width)

    # Centre card (shadow → fill → border)
    cv2.rectangle(panel, (cl + 5, ct + 5), (cr + 5, cb + 5), (0, 0, 0), -1)
    accent_dark = tuple(max(0, c - 30) for c in accent)
    _gradient(panel, cl, ct, cr, cb, accent, accent_dark)
    cv2.rectangle(panel, (cl, ct), (cr, cb), fg, 2)

    # Gesture label inside card
    label = (gesture or "?").upper().replace("_", " ")
    ls = 1.30 if len(label) <= 8 else (1.05 if len(label) <= 12 else 0.80)
    (lw, lh), _ = cv2.getTextSize(label, _FONT, ls, 2)
    while lw > (cr - cl - 16) and ls > 0.5:
        ls -= 0.05
        (lw, lh), _ = cv2.getTextSize(label, _FONT, ls, 2)
    lx = max(cl + 6, (width - lw) // 2)
    ly = (ct + cb) // 2 + lh // 2
    cv2.putText(panel, label, (lx + 3, ly + 3), _FONT, ls, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(panel, label, (lx, ly),          _FONT, ls, fg,        2, cv2.LINE_AA)

    # Corner dots
    for dy in [ct + 16, cb - 16]:
        for dx in [cl + 16, cr - 16]:
            cv2.circle(panel, (dx, dy), 6, (0, 0, 0), -1)
            cv2.circle(panel, (dx, dy), 4, fg,        -1)

    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), accent, 3)


# ── Main renderer ─────────────────────────────────────────────────────────────

def render_meme_panel(
    gesture: Optional[str],
    width:   int = 400,
    height:  int = 480,
) -> np.ndarray:
    """
    Return a (height × width × 3) BGR meme panel for the given gesture.

    With images downloaded:
        [ meme image fills panel ]
        [ darkened top band  +  TOP CAPTION TEXT    ]
        [                                            ]
        [ darkened bot band  +  BOTTOM CAPTION TEXT ]

    Without images (fallback):
        Coloured card with gradient background (original design).
    """
    data  = MEMES.get(gesture, DEFAULT_MEME) if gesture else DEFAULT_MEME
    panel = np.zeros((height, width, 3), dtype=np.uint8)

    img = _load_image(gesture) if gesture else None

    if img is not None:
        # ── Real image path ────────────────────────────────────────────────
        BAND = 58       # height of the top/bottom darkened text bands

        _place_image(panel, img)

        # Darken bands so white text is always readable
        _darken_band(panel, 0,            BAND,   opacity=0.62)
        _darken_band(panel, height - BAND, height, opacity=0.62)

        # Caption text (white + black outline = classic meme style)
        _centered_text(panel, data["top"],    38,          0.80, _WHITE, 2, width)
        _centered_text(panel, data["bottom"], height - 18, 0.80, _WHITE, 2, width)

        # Thin white border around the whole panel
        cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (220, 220, 220), 2)

    else:
        # ── Fallback: no image downloaded yet ─────────────────────────────
        _render_fallback(panel, data, gesture, width, height)

        # Small hint at the bottom of the fallback card
        hint = "run download_memes.py for images"
        (hw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_PLAIN, 0.85, 1)
        hx = max(4, (width - hw) // 2)
        cv2.putText(panel, hint, (hx, height - 4),
                    cv2.FONT_HERSHEY_PLAIN, 0.85, (100, 100, 100), 1, cv2.LINE_AA)

    return panel
