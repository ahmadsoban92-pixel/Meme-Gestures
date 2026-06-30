"""
face_detector.py — Facial expression detection via MediaPipe Face Mesh.

Detects 468 face landmarks and classifies three expressions:
  smile · wink · mouth_open

Techniques used:
  - Eye Aspect Ratio (EAR) for wink detection
  - Inner-lip gap ratio for mouth-open detection
  - Mouth-corner elevation ratio for smile detection
"""

import cv2
import mediapipe as mp
import numpy as np

from config import (
    FACE_DETECTION_CONF, FACE_TRACKING_CONF,
    MOUTH_OPEN_RATIO, EAR_WINK_THRESHOLD, EAR_DIFF_THRESHOLD, SMILE_RATIO,
)


class FaceDetector:
    """Wraps MediaPipe Face Mesh and classifies facial expressions."""

    # ── Key landmark indices (MediaPipe Face Mesh) ────────────────────────────

    # Eye Aspect Ratio points: [outer, upper-outer, upper-inner, inner, lower-inner, lower-outer]
    LEFT_EYE  = [33,  160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    # Mouth
    MOUTH_LEFT  = 61    # left corner of mouth
    MOUTH_RIGHT = 291   # right corner of mouth
    UPPER_LIP   = 13    # centre of upper lip (inner surface)
    LOWER_LIP   = 14    # centre of lower lip (inner surface)

    # Face height reference
    FACE_TOP    = 10    # top of forehead (midline)
    FACE_BOTTOM = 152   # chin tip

    def __init__(self):
        self._mp_face_mesh = mp.solutions.face_mesh
        self._mp_draw      = mp.solutions.drawing_utils

        # FaceMesh constructor — handle older mediapipe without refine_landmarks
        try:
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=FACE_DETECTION_CONF,
                min_tracking_confidence=FACE_TRACKING_CONF,
            )
        except TypeError:
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=FACE_DETECTION_CONF,
                min_tracking_confidence=FACE_TRACKING_CONF,
            )

        self._conn_spec = self._mp_draw.DrawingSpec(
            color=(160, 160, 160), thickness=1, circle_radius=1
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, frame_rgb: np.ndarray):
        """Run MediaPipe Face Mesh on an RGB frame. Returns raw results."""
        return self._face_mesh.process(frame_rgb)

    def classify_expression(self, results) -> "str | None":
        """
        Return the name of the detected facial expression or None.
        Priority: mouth_open > wink > smile
        """
        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark

        # Normalisation: face height in normalised coordinates
        face_h = abs(lm[self.FACE_TOP].y - lm[self.FACE_BOTTOM].y)
        if face_h < 1e-6:
            return None

        # ── Mouth Open 😮 ─────────────────────────────────────────────────────
        mouth_gap = abs(lm[self.UPPER_LIP].y - lm[self.LOWER_LIP].y) / face_h
        if mouth_gap > MOUTH_OPEN_RATIO:
            return "mouth_open"

        # ── Wink 😉 ───────────────────────────────────────────────────────────
        left_ear  = self._ear(self.LEFT_EYE,  lm)
        right_ear = self._ear(self.RIGHT_EYE, lm)

        if abs(left_ear - right_ear) > EAR_DIFF_THRESHOLD:
            if left_ear < EAR_WINK_THRESHOLD or right_ear < EAR_WINK_THRESHOLD:
                return "wink"

        # ── Smile 😊 ──────────────────────────────────────────────────────────
        # Corners elevated above lip centre → positive score
        lip_centre_y  = (lm[self.UPPER_LIP].y + lm[self.LOWER_LIP].y) / 2.0
        corner_avg_y  = (lm[self.MOUTH_LEFT].y + lm[self.MOUTH_RIGHT].y) / 2.0
        smile_score   = (lip_centre_y - corner_avg_y) / face_h
        if smile_score > SMILE_RATIO:
            return "smile"

        return None

    def draw(self, frame: np.ndarray, results) -> None:
        """Overlay face mesh contours on the BGR frame (in-place)."""
        if not results.multi_face_landmarks:
            return
        try:
            for face_lm in results.multi_face_landmarks:
                self._mp_draw.draw_landmarks(
                    frame,
                    face_lm,
                    self._mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self._conn_spec,
                )
        except Exception:
            pass  # gracefully skip if drawing style is unavailable

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _ear(eye_indices: list, lm) -> float:
        """
        Eye Aspect Ratio (EAR):
          (||p2-p6|| + ||p3-p5||) / (2 · ||p1-p4||)
        where p1=outer corner, p4=inner corner, p2/p3=upper, p5/p6=lower.
        """
        p1, p2, p3, p4, p5, p6 = eye_indices

        def d(a, b):
            return float(np.sqrt((lm[a].x - lm[b].x) ** 2 + (lm[a].y - lm[b].y) ** 2))

        vertical = d(p2, p6) + d(p3, p5)
        horizontal = d(p1, p4)
        return vertical / (2.0 * horizontal) if horizontal > 0 else 0.0
