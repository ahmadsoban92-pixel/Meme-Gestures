"""
config.py — Central configuration for Gesture Meme Generator.
Tweak thresholds here if detections feel off on your setup.
"""

# ── Camera ────────────────────────────────────────────────────────────────────
CAM_INDEX  = 0      # Change to 1, 2, … if your webcam isn't the default
CAM_WIDTH  = 640
CAM_HEIGHT = 480

# ── Layout ────────────────────────────────────────────────────────────────────
MEME_PANEL_WIDTH = 400   # Width of the meme panel (pixels)

# ── MediaPipe confidence ──────────────────────────────────────────────────────
HAND_DETECTION_CONF  = 0.75
HAND_TRACKING_CONF   = 0.50
FACE_DETECTION_CONF  = 0.70
FACE_TRACKING_CONF   = 0.50

# ── Gesture stabilisation ─────────────────────────────────────────────────────
HOLD_FRAMES  = 8   # Frames a gesture must persist before it "locks in"
DECAY_FRAMES = 5   # Frames with no detection before the label clears

# ── Hand gesture thresholds ───────────────────────────────────────────────────
# OK sign: thumb-tip ↔ index-tip distance / palm size must be below this
OK_DISTANCE_RATIO = 0.20

# ── Facial expression thresholds ──────────────────────────────────────────────
MOUTH_OPEN_RATIO   = 0.050   # inner-lip gap / face height
EAR_WINK_THRESHOLD = 0.220   # Eye Aspect Ratio below this → eye closed
EAR_DIFF_THRESHOLD = 0.060   # EAR difference between eyes → wink (not blink)
SMILE_RATIO        = 0.010   # corner elevation / face height → smile
