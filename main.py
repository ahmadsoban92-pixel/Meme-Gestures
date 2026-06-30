"""
main.py — Real-Time Gesture Meme Generator
==========================================
Detects hand gestures and facial expressions from your webcam
and overlays a matching meme panel in real time.

Supported detections
────────────────────
  Hand gestures : ✌ Victory  👍 Thumbs Up  👊 Fist  👌 OK  👉 Pointing  🙏 Namaste
  Expressions   : 😊 Smile   😉 Wink       😮 Mouth Open

Controls
────────
  Q     → quit
  S     → toggle skeleton / mesh overlay
  SPACE → freeze the current meme (press again to resume)

Usage
─────
  python main.py
"""

import time
import cv2
import numpy as np

from config import (
    CAM_INDEX, CAM_WIDTH, CAM_HEIGHT, MEME_PANEL_WIDTH,
    HOLD_FRAMES, DECAY_FRAMES,
)
from hand_detector import HandDetector
from face_detector import FaceDetector
from meme_panel   import render_meme_panel


# ── Helpers ────────────────────────────────────────────────────────────────────

def _put(frame, text, pos, scale=0.65, color=(0, 255, 0), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        print(f"❌  Could not open camera at index {CAM_INDEX}.")
        print("    Try changing CAM_INDEX in config.py.")
        return

    hand_det = HandDetector()
    face_det = FaceDetector()

    # ── Gesture stabilisation state ────────────────────────────────────────────
    current_gesture = None   # the "locked-in" gesture shown in the meme
    pending_gesture = None   # candidate waiting to reach HOLD_FRAMES
    hold_count      = 0      # consecutive frames pending_gesture has been seen
    decay_count     = 0      # consecutive frames with no detection

    # ── UI state ───────────────────────────────────────────────────────────────
    show_landmarks = True
    frozen         = False
    frozen_gesture = None

    # ── FPS tracking ───────────────────────────────────────────────────────────
    fps        = 0.0
    prev_time  = time.time()

    print("🚀  Gesture Meme Generator is running.")
    print("    Controls: Q = quit | S = toggle skeleton | SPACE = freeze meme")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️   Webcam read failed. Retrying…")
            continue

        frame = cv2.flip(frame, 1)                        # mirror for natural feel
        h, w  = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Run detectors ──────────────────────────────────────────────────────
        hand_results = hand_det.process(rgb)
        face_results = face_det.process(rgb)

        # Hand gestures take priority over facial expressions
        detected = hand_det.classify_gesture(hand_results)
        if detected is None:
            detected = face_det.classify_expression(face_results)

        # ── Stabilise: hold + decay counters ───────────────────────────────────
        if detected is not None:
            decay_count = 0
            if detected == pending_gesture:
                hold_count += 1
            else:
                pending_gesture = detected
                hold_count      = 1

            if hold_count >= HOLD_FRAMES:
                current_gesture = pending_gesture
        else:
            hold_count  = 0
            decay_count += 1
            if decay_count >= DECAY_FRAMES:
                current_gesture = None
                pending_gesture = None

        # ── Draw landmarks ─────────────────────────────────────────────────────
        if show_landmarks:
            hand_det.draw(frame, hand_results)
            face_det.draw(frame, face_results)

        # ── Meme panel ─────────────────────────────────────────────────────────
        meme_gesture = frozen_gesture if frozen else current_gesture
        panel = render_meme_panel(meme_gesture, MEME_PANEL_WIDTH, h)

        # ── Composite: camera frame + meme panel side by side ──────────────────
        output = np.zeros((h, w + MEME_PANEL_WIDTH, 3), dtype=np.uint8)
        output[:, :w]  = frame
        output[:, w:]  = panel
        cv2.line(output, (w, 0), (w, h), (255, 255, 255), 2)   # divider

        # ── HUD overlays ───────────────────────────────────────────────────────
        # FPS
        now      = time.time()
        fps      = 0.9 * fps + 0.1 / max(now - prev_time, 1e-9)
        prev_time = now
        _put(output, f"FPS: {fps:.0f}", (10, 28), 0.60, (0, 220, 0))

        # Current gesture label
        label = current_gesture.upper().replace("_", " ") if current_gesture else "—"
        _put(output, f"Detected: {label}", (10, 55), 0.72, (0, 255, 0))

        # Freeze notice
        if frozen:
            _put(output, "FROZEN  (SPACE to resume)", (10, 82),
                 0.52, (0, 210, 255))

        # Landmark toggle notice
        if not show_landmarks:
            y = 82 if not frozen else 108
            _put(output, "Skeleton OFF  (S to toggle)", (10, y),
                 0.48, (180, 180, 180))

        # ── Display ────────────────────────────────────────────────────────────
        cv2.imshow("Gesture Meme Generator", output)

        # ── Key handling ───────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting…")
            break
        elif key == ord('s'):
            show_landmarks = not show_landmarks
        elif key == ord(' '):
            frozen = not frozen
            if frozen:
                frozen_gesture = current_gesture
                print(f"Meme frozen on: {frozen_gesture}")
            else:
                print("Meme unfrozen.")

    cap.release()
    cv2.destroyAllWindows()
    print("Bye! 👋")


if __name__ == "__main__":
    main()
