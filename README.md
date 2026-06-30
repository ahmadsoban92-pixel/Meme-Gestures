# 🎭 Real-Time Gesture Meme Generator

A Python computer-vision project that detects **hand gestures** and **facial expressions**
from your webcam and overlays a matching meme panel in real time.

Inspired by [Niladri Das's LinkedIn project](https://www.linkedin.com/in/niladri-das-426746321).

---

## ✨ What it detects

| Category | Gesture / Expression | Meme shown |
|----------|---------------------|------------|
| ✌️ Hand | Victory (peace sign) | "When the code finally works" |
| 👍 Hand | Thumbs Up | "One does not simply build CV on first try" |
| 👊 Hand | Fist | "Challenge Accepted" |
| 👌 Hand | OK Sign | "This is perfectly fine" |
| 👉 Hand | Pointing | "You, yes you — star this repo" |
| 🙏 Hand | Namaste (two open palms) | "Thank you, come again" |
| 😊 Face | Smile | "When you finally understand the code" |
| 😉 Face | Wink | "I see what you did there" |
| 😮 Face | Mouth Open | "Surprised Pikachu face" |

---

## 🚀 Quick start

### 1. Clone / download

```bash
git clone <your-repo-url>
cd gesture-meme-gen
```

### 2. Install dependencies

Python 3.8+ is required.

```bash
pip install -r requirements.txt
```

> **Tip (virtual environment):**
> ```bash
> python -m venv venv
> source venv/bin/activate    # macOS/Linux
> venv\Scripts\activate       # Windows
> pip install -r requirements.txt
> ```

### 3. Run

```bash
python main.py
```

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `Q` | Quit |
| `S` | Toggle hand/face skeleton overlay |
| `SPACE` | Freeze / unfreeze the current meme |

---

## 🗂️ Project structure

```
gesture-meme-gen/
├── main.py            # Webcam loop + compositing
├── hand_detector.py   # MediaPipe Hands wrapper + gesture rules
├── face_detector.py   # MediaPipe Face Mesh wrapper + expression rules
├── meme_panel.py      # OpenCV meme panel renderer
├── config.py          # All thresholds and constants (tweak here)
└── requirements.txt
```

---

## ⚙️ How it works

### Hand gestures — `hand_detector.py`
MediaPipe Hands returns **21 normalised 3-D landmarks** per hand.
Each gesture is classified with simple geometric rules:

| Gesture | Rule |
|---------|------|
| Thumbs Up | Thumb tip above thumb IP; all other fingers curled |
| Victory | Index + middle extended; others curled |
| Fist | No fingers extended |
| Pointing | Index only extended |
| OK | Thumb–index tip distance < 20 % of palm size; middle/ring/pinky up |
| Namaste | Both hands detected as open palms simultaneously |

### Facial expressions — `face_detector.py`
MediaPipe Face Mesh returns **468 face landmarks**.

| Expression | Technique |
|-----------|-----------|
| Mouth Open | Inner-lip gap / face height > threshold |
| Wink | Eye Aspect Ratio (EAR) is low on one side but not both |
| Smile | Mouth corners elevated above lip centre (normalised by face height) |

### Stability — `main.py`
A gesture must be detected for `HOLD_FRAMES` (default 8) consecutive frames before
it "locks in". It is cleared after `DECAY_FRAMES` (default 5) frames with no detection.
This prevents flickering between gestures.

---

## 🔧 Tuning

All thresholds live in **`config.py`**. If detections feel unreliable:

| Problem | Try |
|---------|-----|
| Gesture flickers | Increase `HOLD_FRAMES` |
| Gesture sticks too long | Decrease `DECAY_FRAMES` |
| OK sign not detected | Decrease `OK_DISTANCE_RATIO` |
| Smile triggers too easily | Increase `SMILE_RATIO` |
| Wink not detected | Decrease `EAR_DIFF_THRESHOLD` |
| Mouth open triggers falsely | Increase `MOUTH_OPEN_RATIO` |

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Webcam capture, drawing, display |
| `mediapipe` | Hand and face landmark detection |
| `numpy` | Numerical ops, gradient rendering |

---

## 🪪 Licence
MIT — feel free to use, modify, and share.
