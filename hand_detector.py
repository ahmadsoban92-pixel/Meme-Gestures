"""
hand_detector.py — Hand landmark detection and gesture classification.

Uses MediaPipe Hands to detect 21 landmarks per hand, then applies
simple geometric rules to classify each gesture.

Supported gestures:
  thumbs_up · victory · fist · ok · pointing · namaste (two open palms)
"""

import cv2
import mediapipe as mp
import numpy as np

from config import HAND_DETECTION_CONF, HAND_TRACKING_CONF, OK_DISTANCE_RATIO


class HandDetector:
    """Wraps MediaPipe Hands and provides gesture classification."""

    # ── Landmark index constants ───────────────────────────────────────────
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP,  INDEX_PIP,  INDEX_DIP,  INDEX_TIP  = 5,  6,  7,  8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9,  10, 11, 12
    RING_MCP,   RING_PIP,   RING_DIP,   RING_TIP   = 13, 14, 15, 16
    PINKY_MCP,  PINKY_PIP,  PINKY_DIP,  PINKY_TIP  = 17, 18, 19, 20

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=HAND_DETECTION_CONF,
            min_tracking_confidence=HAND_TRACKING_CONF,
        )
        # Drawing styles
        self._lm_spec   = self._mp_draw.DrawingSpec(color=(0, 255, 0),   thickness=2, circle_radius=3)
        self._conn_spec = self._mp_draw.DrawingSpec(color=(0, 200, 255), thickness=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame_rgb: np.ndarray):
        """Run MediaPipe Hands on an RGB frame. Returns raw results."""
        return self._hands.process(frame_rgb)

    def classify_gesture(self, results) -> "str | None":
        """
        Given MediaPipe Hands results, return the name of the detected gesture
        or None if no clear gesture is found.
        """
        if not results.multi_hand_landmarks:
            return None

        gestures = []
        for hand_lm, handedness in zip(
            results.multi_hand_landmarks, results.multi_handedness
        ):
            g = self._classify_single(hand_lm, handedness)
            if g:
                gestures.append(g)

        if not gestures:
            return None

        # Namaste requires TWO open palms at the same time
        if len(gestures) == 2 and all(g == "open_palm" for g in gestures):
            return "namaste"

        # Otherwise return the primary (first) hand's gesture
        return gestures[0]

    def draw(self, frame: np.ndarray, results) -> None:
        """Overlay hand skeleton landmarks on the BGR frame (in-place)."""
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame, hand_lm,
                    self._mp_hands.HAND_CONNECTIONS,
                    self._lm_spec, self._conn_spec,
                )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _dist(lm, a: int, b: int) -> float:
        """Euclidean distance between two normalised landmarks."""
        return float(np.sqrt((lm[a].x - lm[b].x) ** 2 + (lm[a].y - lm[b].y) ** 2))

    def _fingers_up(self, lm, label: str) -> list:
        """
        Return a 5-element bool list [thumb, index, middle, ring, pinky]
        where True means the finger is extended.
        """
        # Thumb uses x-axis (mirrored for left vs right hand)
        if label == "Right":
            thumb = lm[self.THUMB_TIP].x < lm[self.THUMB_IP].x
        else:
            thumb = lm[self.THUMB_TIP].x > lm[self.THUMB_IP].x

        # Four fingers: tip y < pip y means extended (y increases downward)
        index  = lm[self.INDEX_TIP].y  < lm[self.INDEX_PIP].y
        middle = lm[self.MIDDLE_TIP].y < lm[self.MIDDLE_PIP].y
        ring   = lm[self.RING_TIP].y   < lm[self.RING_PIP].y
        pinky  = lm[self.PINKY_TIP].y  < lm[self.PINKY_PIP].y

        return [thumb, index, middle, ring, pinky]

    def _classify_single(self, hand_lm, handedness) -> "str | None":
        """Classify a single hand's gesture."""
        lm    = hand_lm.landmark
        label = handedness.classification[0].label   # "Left" or "Right"
        f     = self._fingers_up(lm, label)
        thumb, index, middle, ring, pinky = f

        # ── Thumbs Up ─────────────────────────────────────────────────────────
        if thumb and not index and not middle and not ring and not pinky:
            return "thumbs_up"

        # ── Fist ──────────────────────────────────────────────────────────────
        if not any(f):
            return "fist"

        # ── Victory / Peace ✌️ ────────────────────────────────────────────────
        if not thumb and index and middle and not ring and not pinky:
            return "victory"

        # ── Pointing ──────────────────────────────────────────────────────────
        if not thumb and index and not middle and not ring and not pinky:
            return "pointing"

        # ── OK Sign 👌 ────────────────────────────────────────────────────────
        # Thumb + index tips form a circle; other three fingers extended
        palm_size = self._dist(lm, self.WRIST, self.MIDDLE_MCP)
        tip_dist  = self._dist(lm, self.THUMB_TIP, self.INDEX_TIP)
        if palm_size > 0 and (tip_dist / palm_size) < OK_DISTANCE_RATIO:
            if middle and ring and pinky:
                return "ok"

        # ── Open Palm (prerequisite for Namaste) ──────────────────────────────
        if thumb and index and middle and ring and pinky:
            return "open_palm"

        return None
