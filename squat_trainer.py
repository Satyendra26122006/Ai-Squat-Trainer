"""
╔══════════════════════════════════════════════════════════════════╗
║           AI SQUAT TRAINER - FULL SKELETON TRACKING             ║
║     Live Camera | Multi-Angle Analysis | Real-Time Feedback     ║
╚══════════════════════════════════════════════════════════════════╝

REQUIREMENTS (install before running):
    pip install opencv-python mediapipe>=0.10 numpy

RUN:
    python squat_trainer.py

CONTROLS:
    Q / ESC  - Quit
    R        - Reset counter
    S        - Toggle side-panel stats
    C        - Calibrate (capture standing pose as baseline)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math
import sys
import os
import csv
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Dict

# ─────────────────────────────────────────────────────────────────
#  MEDIAPIPE VERSION-SAFE SETUP
#  Supports both mediapipe <0.10 (mp.solutions) and >=0.10 (tasks API)
# ─────────────────────────────────────────────────────────────────

def _get_mp_version() -> Tuple[int, int]:
    try:
        parts = mp.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return (0, 9)

_MP_MAJOR, _MP_MINOR = _get_mp_version()
_USE_LEGACY_API = (_MP_MAJOR == 0 and _MP_MINOR < 10)

print(f"[INFO] MediaPipe {mp.__version__} detected — "
      f"using {'legacy solutions' if _USE_LEGACY_API else 'new tasks'} API")


def _build_pose():
    """Build a pose estimator compatible with the installed MediaPipe version."""
    if _USE_LEGACY_API:
        # Old API: mp.solutions.pose  (mediapipe < 0.10)
        mp_pose = mp.solutions.pose
        return mp_pose.Pose(
            model_complexity=2,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        ), "legacy"
    else:
        # New API: mediapipe >= 0.10 — mp.solutions.pose still works in 0.10.x
        # but mp.solutions is removed in some 0.10+ builds.
        # Try solutions first, fall back gracefully.
        try:
            mp_pose = mp.solutions.pose
            pose = mp_pose.Pose(
                model_complexity=2,
                enable_segmentation=False,
                smooth_landmarks=True,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            return pose, "legacy"
        except AttributeError:
            pass

        # Full new Tasks API path
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
            import urllib.request, os, tempfile

            MODEL_URL = (
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_heavy/float16/latest/"
                "pose_landmarker_heavy.task"
            )
            model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_heavy.task")
            if not os.path.exists(model_path):
                print("[INFO] Downloading high-precision pose model (~25 MB)…")
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print("[INFO] Download complete.")

            base_opts = mp_python.BaseOptions(model_asset_path=model_path)
            opts = mp_vision.PoseLandmarkerOptions(
                base_options=base_opts,
                output_segmentation_masks=False,
                num_poses=1,
                min_pose_detection_confidence=0.65,
                min_pose_presence_confidence=0.65,
                min_tracking_confidence=0.65,
                running_mode=mp_vision.RunningMode.VIDEO,
            )
            return mp_vision.PoseLandmarker.create_from_options(opts), "tasks"
        except Exception as e:
            print(f"[ERROR] Could not initialise MediaPipe Tasks API: {e}")
            print("[HINT] Try:  pip install mediapipe --upgrade")
            sys.exit(1)


class _PoseWrapper:
    """
    Unified wrapper — exposes a single .process(rgb_frame) -> result
    regardless of which API is used underneath.
    result.pose_landmarks  → list of landmark objects with .x .y .z
    """

    def __init__(self):
        self._pose, self._mode = _build_pose()
        self._frame_ts = 0  # millisecond counter for Tasks API

    def process(self, rgb_frame):
        if self._mode == "legacy":
            return self._pose.process(rgb_frame)
        else:
            # Tasks API needs an mp.Image and a timestamp
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            self._frame_ts += 33  # ~30 fps
            result = self._pose.detect_for_video(mp_image, self._frame_ts)
            # Wrap in a duck-typed object to match legacy API
            return _TasksResultWrapper(result)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        try:
            self._pose.close()
        except Exception:
            pass


class _TasksResultWrapper:
    """Makes Tasks API result look like legacy PoseLandmarker result."""
    def __init__(self, result):
        self._result = result

    @property
    def pose_landmarks(self):
        if not self._result.pose_landmarks:
            return None
        # Tasks API returns NormalizedLandmark objects — wrap in a list-like
        return _LandmarkListWrapper(self._result.pose_landmarks[0])


class _LandmarkListWrapper:
    def __init__(self, landmarks):
        self._lm = landmarks

    @property
    def landmark(self):
        return self._lm  # Already a list of NormalizedLandmark

    def __getitem__(self, idx):
        return self._lm[idx]

# ─────────────────────────────────────────────────────────────────
#  ENUMS & DATA CLASSES
# ─────────────────────────────────────────────────────────────────

class SquatPhase(Enum):
    STANDING   = "STANDING"
    DESCENDING = "DESCENDING"
    BOTTOM     = "BOTTOM"
    ASCENDING  = "ASCENDING"


class FormError(Enum):
    KNEE_CAVE       = "Knees caving inward"
    FORWARD_LEAN    = "Excessive forward lean"
    HEEL_RISE       = "Heels rising"
    SHALLOW_SQUAT   = "Squat not deep enough"
    KNEE_OVER_TOE   = "Knees too far over toes"
    BACK_ROUND      = "Back rounding"
    UNEVEN_HIPS     = "Hips uneven"
    NECK_FORWARD    = "Head/neck position"


@dataclass
class AnglesSnapshot:
    """All key angles at a single frame."""
    left_knee:        float = 0.0
    right_knee:       float = 0.0
    left_hip:         float = 0.0
    right_hip:        float = 0.0
    left_ankle:       float = 0.0
    right_ankle:      float = 0.0
    left_back:        float = 0.0   # hip flexion / back posture
    right_back:       float = 0.0   # hip flexion / back posture
    knee_over_toe:    float = 0.0   # normalised horizontal offset
    torso_lean:       float = 0.0   # degrees from vertical
    hip_level_diff:   float = 0.0   # |left_hip_y - right_hip_y| normalised
    knee_width_ratio: float = 0.0   # knee-width / hip-width ratio
    neck_angle:       float = 0.0   # head forward lean
    shoulder_level:   float = 0.0   # shoulder tilt


@dataclass
class SquatRep:
    bottom_angles:  AnglesSnapshot = field(default_factory=AnglesSnapshot)
    errors:         List[FormError] = field(default_factory=list)
    score:          int = 0         # 0-100
    depth_reached:  float = 0.0    # lowest knee angle
    timestamp:      float = field(default_factory=time.time)


# ─────────────────────────────────────────────────────────────────
#  CONSTANTS – Biomechanical thresholds
# ─────────────────────────────────────────────────────────────────

THRESHOLDS = {
    # Knee angle in degrees (0° = fully bent, 180° = fully straight)
    "squat_start":           155,   # below this → descending
    "squat_bottom":          100,   # below this → bottom phase (parallel ~90°)
    "squat_parallel":         95,   # ideal depth
    "squat_deep":             80,   # ATG depth bonus

    # Form thresholds
    "torso_lean_max":         45,   # degrees from vertical before warning
    "knee_cave_ratio_min":   0.75,  # knee-width / hip-width – below = cave
    "knee_over_toe_max":     0.15,  # normalised – tibia angle from ankle
    "hip_level_max":         0.04,  # normalised y-diff between hips
    "ankle_dorsiflexion_min": 65,   # ankle angle minimum

    # Smoothing
    "angle_smooth_frames":    5,
}

IDEAL_ANGLES = {
    "knee_bottom":   90,   # parallel squat
    "hip_bottom":    90,
    "ankle_bottom":  70,
    "torso_lean":    20,   # slight forward lean is natural
}

DATASET_HEADERS = [
    "timestamp", "phase", "left_knee", "right_knee", "left_hip", "right_hip",
    "left_ankle", "right_ankle", "left_back", "right_back", "knee_over_toe",
    "torso_lean", "neck_angle", "knee_width_ratio",
    "hip_level_diff", "shoulder_level", "rep_count", "errors", "score"
]
DATASET_FILE = "squat_dataset.csv"
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 540

# ─────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (BGR)
# ─────────────────────────────────────────────────────────────────
C = {
    "bg":         (15,  15,  25),
    "green":      (50, 220, 100),
    "red":        (50,  50, 220),
    "yellow":     (0,  200, 220),
    "cyan":       (220, 200,  50),
    "white":      (240, 240, 240),
    "grey":       (120, 120, 130),
    "dark_grey":  (40,  40,  50),
    "orange":     (30, 140, 220),
    "purple":     (200,  80, 180),
    "accent":     (0,  180, 255),
    "good":       (80, 200,  80),
    "warn":       (0,  165, 255),
    "bad":        (60,  60, 230),
}

# ─────────────────────────────────────────────────────────────────
#  UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def calc_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the angle ABC (at vertex B) in degrees.
    Works in both 2-D and 3-D.
    """
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def calc_angle_vertical(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of segment a→b from the vertical axis (in degrees)."""
    vec = b - a
    vertical = np.array([0, 1, 0]) if len(vec) == 3 else np.array([0, 1])
    norm_vec = np.linalg.norm(vec)
    if norm_vec < 1e-6:
        return 0.0
    cosine = np.dot(vec, vertical) / norm_vec
    cosine = np.clip(cosine, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def landmark_to_np(landmark, w: int, h: int, use_z: bool = True) -> np.ndarray:
    if use_z:
        return np.array([landmark.x * w, landmark.y * h, landmark.z * w])
    return np.array([landmark.x * w, landmark.y * h])


def smooth_values(buffer: deque, new_val: float) -> float:
    buffer.append(new_val)
    return float(np.mean(buffer))


def score_angle(actual: float, ideal: float, tolerance: float = 15.0) -> float:
    """Returns 0-1 score based on closeness to ideal angle."""
    diff = abs(actual - ideal)
    return max(0.0, 1.0 - diff / (tolerance * 2))


def draw_rounded_rect(img, pt1, pt2, color, thickness, radius=8):
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def draw_filled_rounded_rect(img, pt1, pt2, color, radius=8, alpha=0.6):
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def angle_color(angle: float, ideal: float, tolerance: float = 20.0) -> tuple:
    diff = abs(angle - ideal)
    if diff < tolerance * 0.5:
        return C["good"]
    elif diff < tolerance:
        return C["warn"]
    return C["bad"]


# ─────────────────────────────────────────────────────────────────
#  ANGLE GAUGE WIDGET
# ─────────────────────────────────────────────────────────────────

def draw_angle_gauge(img, cx: int, cy: int, angle: float,
                     label: str, ideal: float, radius: int = 28):
    color = angle_color(angle, ideal)
    # Background arc
    cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, 300, C["dark_grey"], 3)
    # Value arc (map 0-180 to 0-300 degrees sweep)
    sweep = int(min(300, (angle / 180.0) * 300))
    cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, sweep, color, 3)
    # Centre text
    cv2.putText(img, f"{int(angle)}", (cx - 14, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(img, label, (cx - len(label)*4, cy + radius + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, C["grey"], 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
#  DEPTH METER
# ─────────────────────────────────────────────────────────────────

def draw_depth_meter(img, x: int, y: int, knee_angle: float):
    """Vertical bar showing squat depth (100° = parallel = green zone)."""
    h, w_bar = 160, 16
    # Background
    draw_filled_rounded_rect(img, (x, y), (x + w_bar, y + h), C["dark_grey"], radius=4, alpha=0.8)
    # Parallel marker line
    parallel_y = int(y + h * (1 - 90 / 180))
    cv2.line(img, (x - 4, parallel_y), (x + w_bar + 4, parallel_y), C["cyan"], 1)
    cv2.putText(img, "PAR", (x + w_bar + 5, parallel_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, C["cyan"], 1, cv2.LINE_AA)
    # Fill
    fill_ratio = 1.0 - min(1.0, max(0.0, knee_angle / 180.0))
    fill_h = int(h * fill_ratio)
    if fill_h > 0:
        col = C["good"] if knee_angle <= 95 else (C["warn"] if knee_angle <= 115 else C["bad"])
        draw_filled_rounded_rect(img, (x, y + h - fill_h), (x + w_bar, y + h),
                                 col, radius=4, alpha=0.9)
    # Label
    cv2.putText(img, "DEPTH", (x - 2, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, C["grey"], 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
#  SKELETON OVERLAY
# ─────────────────────────────────────────────────────────────────

# MediaPipe landmark indices
LM = {
    "nose":           0,
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
    "left_heel":      29, "right_heel":     30,
    "left_foot_index":31, "right_foot_index":32,
}

SKELETON_CONNECTIONS = [
    # Torso
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    # Left leg
    ("left_hip",   "left_knee"),
    ("left_knee",  "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_foot_index"),
    # Right leg
    ("right_hip",   "right_knee"),
    ("right_knee",  "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_foot_index"),
    # Arms
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    # Neck
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
]


def draw_skeleton(img, landmarks, w: int, h: int,
                  angles: AnglesSnapshot, phase: SquatPhase):
    """Draw colour-coded skeleton with joint circles."""
    pts: Dict[str, Tuple[int, int]] = {}
    for name, idx in LM.items():
        lm = landmarks[idx]
        pts[name] = (int(lm.x * w), int(lm.y * h))

    # Connections
    phase_color = {
        SquatPhase.STANDING:   C["green"],
        SquatPhase.DESCENDING: C["yellow"],
        SquatPhase.BOTTOM:     C["cyan"],
        SquatPhase.ASCENDING:  C["orange"],
    }[phase]

    for (a, b) in SKELETON_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], phase_color, 2, cv2.LINE_AA)

    # Joint circles
    joint_radii = {"left_knee": 7, "right_knee": 7,
                   "left_hip": 6,  "right_hip": 6,
                   "left_ankle": 5, "right_ankle": 5}
    for name, pt in pts.items():
        r = joint_radii.get(name, 4)
        cv2.circle(img, pt, r + 2, (0, 0, 0), -1)
        cv2.circle(img, pt, r, C["white"], -1)

    # Angle arcs at key joints
    _draw_joint_angle(img, pts, "left_hip",   "left_shoulder",  "left_knee",
                      angles.left_hip,  IDEAL_ANGLES["hip_bottom"])
    _draw_joint_angle(img, pts, "right_hip",  "right_shoulder", "right_knee",
                      angles.right_hip, IDEAL_ANGLES["hip_bottom"])
    _draw_joint_angle(img, pts, "left_knee",  "left_hip",  "left_ankle",
                      angles.left_knee, IDEAL_ANGLES["knee_bottom"])
    _draw_joint_angle(img, pts, "right_knee", "right_hip", "right_ankle",
                      angles.right_knee, IDEAL_ANGLES["knee_bottom"])
    _draw_joint_angle(img, pts, "left_ankle", "left_knee", "left_foot_index",
                      angles.left_ankle, IDEAL_ANGLES["ankle_bottom"])

    return pts


def _draw_joint_angle(img, pts, vertex: str, p1: str, p2: str,
                      angle: float, ideal: float):
    if vertex not in pts or p1 not in pts or p2 not in pts:
        return
    col = angle_color(angle, ideal, tolerance=25)
    cx, cy = pts[vertex]
    # Small floating label
    draw_filled_rounded_rect(img, (cx + 8, cy - 14), (cx + 48, cy + 2),
                             (0, 0, 0), radius=3, alpha=0.5)
    cv2.putText(img, f"{int(angle)}", (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
#  HUD OVERLAY
# ─────────────────────────────────────────────────────────────────

def draw_hud(img, angles: AnglesSnapshot, phase: SquatPhase,
             rep_count: int, errors: List[FormError],
             rep_score: int, fps: float, show_stats: bool,
             rep_history: List[SquatRep], calibrated: bool, recording: bool):
    h, w = img.shape[:2]

    # ── Top bar ────────────────────────────────────────────────
    draw_filled_rounded_rect(img, (0, 0), (w, 52), C["bg"], radius=0, alpha=0.75)

    phase_colors = {
        SquatPhase.STANDING:   C["green"],
        SquatPhase.DESCENDING: C["yellow"],
        SquatPhase.BOTTOM:     C["cyan"],
        SquatPhase.ASCENDING:  C["orange"],
    }
    phase_col = phase_colors[phase]

    cv2.putText(img, "AI SQUAT TRAINER", (12, 32),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, C["accent"], 1, cv2.LINE_AA)

    # Phase badge
    px = 240
    draw_filled_rounded_rect(img, (px, 8), (px + 140, 44), phase_col, radius=6, alpha=0.35)
    draw_rounded_rect(img, (px, 8), (px + 140, 44), phase_col, 1, radius=6)
    cv2.putText(img, phase.value, (px + 8, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, phase_col, 1, cv2.LINE_AA)

    # Rep counter
    cv2.putText(img, f"REPS: {rep_count}", (w - 160, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, C["white"], 1, cv2.LINE_AA)

    # FPS
    cv2.putText(img, f"{fps:.0f} fps", (w - 80, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C["grey"], 1, cv2.LINE_AA)

    # Calibration badge
    if calibrated:
        cv2.putText(img, "✓ CALIBRATED", (w - 260, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C["good"], 1, cv2.LINE_AA)

    if recording:
        cv2.putText(img, "● DATA RECORDING", (w - 320, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, C["orange"], 1, cv2.LINE_AA)

    gx, gy = 36, 90
    gauge_data = [
        (angles.left_knee,   "L.KNEE",  IDEAL_ANGLES["knee_bottom"]),
        (angles.right_knee,  "R.KNEE",  IDEAL_ANGLES["knee_bottom"]),
        (angles.left_hip,    "L.HIP",   IDEAL_ANGLES["hip_bottom"]),
        (angles.right_hip,   "R.HIP",   IDEAL_ANGLES["hip_bottom"]),
        (angles.left_ankle,  "L.ANKL",  IDEAL_ANGLES["ankle_bottom"]),
        (angles.right_ankle, "R.ANKL",  IDEAL_ANGLES["ankle_bottom"]),
        (angles.torso_lean,  "TORSO",   IDEAL_ANGLES["torso_lean"]),
        (angles.neck_angle,  "NECK",    15.0),
    ]
    draw_filled_rounded_rect(img, (0, 65), (80, gy + len(gauge_data) * 75 + 10),
                             C["bg"], radius=0, alpha=0.7)
    for i, (ang, lbl, ideal) in enumerate(gauge_data):
        draw_angle_gauge(img, gx, gy + i * 72, ang, lbl, ideal, radius=25)

    # ── Right panel – depth meter + symmetry ───────────────────
    rx = w - 45
    avg_knee = (angles.left_knee + angles.right_knee) / 2
    draw_depth_meter(img, rx - 16, 65, avg_knee)

    # Symmetry bar
    sym_y = 260
    sym_diff = abs(angles.left_knee - angles.right_knee)
    sym_score = max(0, 1.0 - sym_diff / 20.0)
    sym_col = C["good"] if sym_score > 0.8 else (C["warn"] if sym_score > 0.5 else C["bad"])
    draw_filled_rounded_rect(img, (rx - 18, sym_y), (rx + 8, sym_y + 60), C["dark_grey"],
                             radius=4, alpha=0.8)
    fill_h2 = int(60 * sym_score)
    if fill_h2 > 0:
        draw_filled_rounded_rect(img, (rx - 18, sym_y + 60 - fill_h2),
                                 (rx + 8, sym_y + 60), sym_col, radius=4, alpha=0.9)
    cv2.putText(img, "SYM", (rx - 18, sym_y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, C["grey"], 1, cv2.LINE_AA)

    # ── Bottom bar – errors & score ─────────────────────────────
    by = h - 95
    draw_filled_rounded_rect(img, (0, by), (w, h), C["bg"], radius=0, alpha=0.75)

    if errors:
        ex = 10
        cv2.putText(img, "⚠ FORM ERRORS:", (ex, by + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["bad"], 1, cv2.LINE_AA)
        for j, err in enumerate(errors[:4]):  # max 4 errors shown
            cv2.putText(img, f"• {err.value}", (ex + j * (w // 4), by + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["warn"], 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "✓  FORM LOOKS GOOD", (10, by + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["good"], 1, cv2.LINE_AA)

    # Score bar
    score_x = w - 200
    score_label = f"SCORE: {rep_score}/100"
    score_col = C["good"] if rep_score >= 80 else (C["warn"] if rep_score >= 55 else C["bad"])
    cv2.putText(img, score_label, (score_x, by + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, score_col, 1, cv2.LINE_AA)
    # Score progress bar
    bar_w = 180
    cv2.rectangle(img, (score_x, by + 30), (score_x + bar_w, by + 42), C["dark_grey"], -1)
    fill = int(bar_w * rep_score / 100)
    cv2.rectangle(img, (score_x, by + 30), (score_x + fill, by + 42), score_col, -1)

    # Controls hint
    cv2.putText(img, "Q=quit  R=reset  C=calibrate  S=stats",
                (10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, C["grey"], 1, cv2.LINE_AA)

    # ── Stats panel (toggle with S) ────────────────────────────
    if show_stats and rep_history:
        _draw_stats_panel(img, rep_history, w, h)


def _draw_stats_panel(img, rep_history: List[SquatRep], w: int, h: int):
    pw, ph = 220, min(len(rep_history) * 26 + 50, 200)
    px, py = w // 2 - pw // 2, 60
    draw_filled_rounded_rect(img, (px, py), (px + pw, py + ph), C["bg"], radius=8, alpha=0.88)
    draw_rounded_rect(img, (px, py), (px + pw, py + ph), C["accent"], 1, radius=8)
    cv2.putText(img, "REP HISTORY", (px + 12, py + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C["accent"], 1, cv2.LINE_AA)
    for i, rep in enumerate(reversed(rep_history[-6:])):
        col = C["good"] if rep.score >= 80 else (C["warn"] if rep.score >= 55 else C["bad"])
        cv2.putText(img, f"Rep {len(rep_history) - i}:  {rep.score}/100  "
                        f"knee={int(rep.depth_reached)}°",
                    (px + 12, py + 42 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────
#  SQUAT ANALYSER  (core logic)
# ─────────────────────────────────────────────────────────────────

class SquatAnalyser:
    def __init__(self):
        self.pose = _PoseWrapper()   # version-safe wrapper

        # Smooth buffers
        n = THRESHOLDS["angle_smooth_frames"]
        self._bufs = {k: deque(maxlen=n) for k in [
            "lk", "rk", "lh", "rh", "la", "ra", "lb", "rb", "ko", "torso", "neck",
        ]}

        self.phase          = SquatPhase.STANDING
        self.rep_count      = 0
        self.rep_history:   List[SquatRep] = []
        self.current_rep    = SquatRep()
        self._min_knee_ang  = 180.0   # track deepest point this rep
        self._calibrated    = False
        self._baseline_hip_y: Optional[float] = None  # standing hip height (normalised)
        self._live_score    = 100
        self._live_errors:  List[FormError] = []
        self._dataset_file = DATASET_FILE
        self._recording = False

        # FPS tracking
        self._fps_buf = deque(maxlen=20)
        self._prev_t  = time.time()

    def calibrate(self, landmarks):
        """Store standing baseline from current pose."""
        lh = landmarks[LM["left_hip"]]
        rh = landmarks[LM["right_hip"]]
        self._baseline_hip_y = (lh.y + rh.y) / 2
        self._calibrated = True

    def _get_np(self, landmarks, name: str, w: int, h: int) -> np.ndarray:
        lm = landmarks[LM[name]]
        return np.array([lm.x * w, lm.y * h, lm.z * w])

    def process(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]

        # FPS
        now = time.time()
        dt = now - self._prev_t
        self._prev_t = now
        if dt > 0:
            self._fps_buf.append(1.0 / dt)
        fps = float(np.mean(self._fps_buf)) if self._fps_buf else 0

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.pose.process(rgb)
        rgb.flags.writeable = True

        if not results.pose_landmarks:
            cv2.putText(frame, "NO PERSON DETECTED - STEP INTO FRAME",
                        (w // 2 - 200, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, C["bad"], 2, cv2.LINE_AA)
            return frame

        # landmark list — works for both legacy (.landmark) and Tasks (list directly)
        raw_lm = results.pose_landmarks
        if hasattr(raw_lm, 'landmark'):
            lm = raw_lm.landmark
        else:
            lm = raw_lm

        # ── Extract 3-D landmark positions ─────────────────────
        def g(name): return self._get_np(lm, name, w, h)

        l_shoulder  = g("left_shoulder");  r_shoulder  = g("right_shoulder")
        l_hip       = g("left_hip");       r_hip       = g("right_hip")
        l_knee      = g("left_knee");      r_knee      = g("right_knee")
        l_ankle     = g("left_ankle");     r_ankle     = g("right_ankle")
        l_heel      = g("left_heel");      r_heel      = g("right_heel")
        l_foot      = g("left_foot_index");r_foot      = g("right_foot_index")
        nose        = g("nose")

        mid_shoulder = (l_shoulder + r_shoulder) / 2
        mid_hip      = (l_hip + r_hip) / 2

        # ── Calculate all angles ────────────────────────────────
        raw_lk = calc_angle(l_hip,       l_knee,   l_ankle)
        raw_rk = calc_angle(r_hip,       r_knee,   r_ankle)
        raw_lh = calc_angle(l_shoulder,  l_hip,    l_knee)
        raw_rh = calc_angle(r_shoulder,  r_hip,    r_knee)
        raw_la = calc_angle(l_knee,      l_ankle,  l_foot)
        raw_ra = calc_angle(r_knee,      r_ankle,  r_foot)
        raw_lb = calc_angle(l_shoulder,  l_hip,    l_knee)
        raw_rb = calc_angle(r_shoulder,  r_hip,    r_knee)
        raw_knee_over_toe_l = abs(l_knee[0] - l_ankle[0]) / max(abs(l_hip[0] - l_ankle[0]), 1e-6)
        raw_knee_over_toe_r = abs(r_knee[0] - r_ankle[0]) / max(abs(r_hip[0] - r_ankle[0]), 1e-6)
        raw_knee_over_toe = max(raw_knee_over_toe_l, raw_knee_over_toe_r)
        raw_torso = calc_angle_vertical(mid_hip, mid_shoulder)
        raw_neck  = calc_angle_vertical(mid_shoulder, nose)

        # Smoothed
        angles = AnglesSnapshot(
            left_knee      = smooth_values(self._bufs["lk"], raw_lk),
            right_knee     = smooth_values(self._bufs["rk"], raw_rk),
            left_hip       = smooth_values(self._bufs["lh"], raw_lh),
            right_hip      = smooth_values(self._bufs["rh"], raw_rh),
            left_ankle     = smooth_values(self._bufs["la"], raw_la),
            right_ankle    = smooth_values(self._bufs["ra"], raw_ra),
            left_back      = smooth_values(self._bufs["lb"], raw_lb),
            right_back     = smooth_values(self._bufs["rb"], raw_rb),
            knee_over_toe  = smooth_values(self._bufs["ko"], raw_knee_over_toe),
            torso_lean     = smooth_values(self._bufs["torso"], raw_torso),
            neck_angle     = smooth_values(self._bufs["neck"],  raw_neck),
        )

        # Hip symmetry (normalised)
        hip_w = abs(lm[LM["left_hip"]].x - lm[LM["right_hip"]].x)
        angles.hip_level_diff = abs(lm[LM["left_hip"]].y - lm[LM["right_hip"]].y) / max(hip_w, 1e-4)

        # Knee width vs hip width ratio (knees should be ~ hip-width or wider)
        knee_w = abs(lm[LM["left_knee"]].x - lm[LM["right_knee"]].x)
        angles.knee_width_ratio = knee_w / max(hip_w, 1e-4)

        # Shoulder tilt
        angles.shoulder_level = abs(lm[LM["left_shoulder"]].y - lm[LM["right_shoulder"]].y)

        # Shoulder level
        angles.shoulder_level = abs(lm[LM["left_shoulder"]].y - lm[LM["right_shoulder"]].y)

        avg_knee = (angles.left_knee + angles.right_knee) / 2

        # ── Phase state machine ─────────────────────────────────
        prev_phase = self.phase

        if self.phase == SquatPhase.STANDING:
            if avg_knee < THRESHOLDS["squat_start"]:
                self.phase = SquatPhase.DESCENDING
                self._min_knee_ang = avg_knee
                self.current_rep = SquatRep()

        elif self.phase == SquatPhase.DESCENDING:
            self._min_knee_ang = min(self._min_knee_ang, avg_knee)
            if avg_knee < THRESHOLDS["squat_bottom"]:
                self.phase = SquatPhase.BOTTOM
                self.current_rep.depth_reached = self._min_knee_ang
                self.current_rep.bottom_angles = angles

        elif self.phase == SquatPhase.BOTTOM:
            if avg_knee > THRESHOLDS["squat_bottom"] + 10:
                self.phase = SquatPhase.ASCENDING

        elif self.phase == SquatPhase.ASCENDING:
            if avg_knee > THRESHOLDS["squat_start"] - 10:
                self.phase = SquatPhase.STANDING
                # Finalise rep
                errors, score = self._evaluate_rep(angles)
                self.current_rep.errors = errors
                self.current_rep.score  = score
                self.rep_history.append(self.current_rep)
                self.rep_count += 1

        # ── Live form check ─────────────────────────────────────
        self._live_errors, self._live_score = self._live_form_check(angles, avg_knee)
        self._log_dataset(angles, avg_knee)

        # ── Draw everything ─────────────────────────────────────
        # Dark overlay for aesthetics
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 20), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # Skeleton
        draw_skeleton(frame, lm, w, h, angles, self.phase)

        # HUD
        draw_hud(frame, angles, self.phase,
                 self.rep_count, self._live_errors,
                 self._live_score, fps,
                 show_stats=self._show_stats,
                 rep_history=self.rep_history,
                 calibrated=self._calibrated,
                 recording=self._recording)

        return frame

    def _live_form_check(self, a: AnglesSnapshot, avg_knee: float
                         ) -> Tuple[List[FormError], int]:
        """Check form in real-time, return errors and live score."""
        errors = []
        score = 100

        # Only check form during squat (not standing idle)
        if avg_knee > THRESHOLDS["squat_start"]:
            return errors, score

        # 1. Knee cave
        if a.knee_width_ratio < THRESHOLDS["knee_cave_ratio_min"]:
            errors.append(FormError.KNEE_CAVE)
            score -= 20

        # 2. Excessive forward lean
        if a.torso_lean > THRESHOLDS["torso_lean_max"]:
            errors.append(FormError.FORWARD_LEAN)
            score -= 15

        # 3. Back rounding / poor hip posture
        if min(a.left_back, a.right_back) < 145:
            errors.append(FormError.BACK_ROUND)
            score -= 15

        # 4. Knee over toe
        if a.knee_over_toe > 0.18:
            errors.append(FormError.KNEE_OVER_TOE)
            score -= 12

        # 5. Shallow squat (only warn below descending threshold)
        if avg_knee < THRESHOLDS["squat_bottom"] + 20 and \
           avg_knee > THRESHOLDS["squat_bottom"]:
            errors.append(FormError.SHALLOW_SQUAT)
            score -= 10

        # 6. Uneven hips
        if a.hip_level_diff > THRESHOLDS["hip_level_max"]:
            errors.append(FormError.UNEVEN_HIPS)
            score -= 10

        # 7. Neck forward
        if a.neck_angle > 35:
            errors.append(FormError.NECK_FORWARD)
            score -= 5

        # 8. Ankle dorsiflexion
        avg_ankle = (a.left_ankle + a.right_ankle) / 2
        if avg_ankle < THRESHOLDS["ankle_dorsiflexion_min"]:
            errors.append(FormError.HEEL_RISE)
            score -= 10

        # Symmetry penalty
        sym_diff = abs(a.left_knee - a.right_knee)
        if sym_diff > 12:
            score -= int(sym_diff / 1.5)

        return errors, max(0, score)

    def _evaluate_rep(self, a: AnglesSnapshot) -> Tuple[List[FormError], int]:
        """Score a completed rep."""
        errors, base = self._live_form_check(a, self.current_rep.depth_reached)

        # Depth bonus
        depth = self.current_rep.depth_reached
        if depth <= THRESHOLDS["squat_deep"]:
            base = min(100, base + 10)   # ATG bonus
        elif depth > THRESHOLDS["squat_bottom"] + 20:
            errors.append(FormError.SHALLOW_SQUAT)
            base -= 15

        # Angle deviation from ideal
        knee_score  = score_angle((a.left_knee + a.right_knee) / 2,
                                   IDEAL_ANGLES["knee_bottom"], 18)
        hip_score   = score_angle((a.left_hip + a.right_hip) / 2,
                                   IDEAL_ANGLES["hip_bottom"], 18)
        torso_score = score_angle(a.torso_lean, IDEAL_ANGLES["torso_lean"], 12)
        back_score  = score_angle(min(a.left_back, a.right_back), 170, 12)
        toe_penalty = 1.0 - min(1.0, a.knee_over_toe / 0.25)

        combined = (knee_score * 0.35 + hip_score * 0.25 + torso_score * 0.2 + back_score * 0.2) * toe_penalty
        final = int(base * 0.55 + combined * 100 * 0.45)
        return errors, max(0, min(100, final))
    def _log_dataset(self, angles: AnglesSnapshot, avg_knee: float):
        if not self._recording:
            return

        row = [
            time.time(), self.phase.value,
            angles.left_knee, angles.right_knee,
            angles.left_hip, angles.right_hip,
            angles.left_ankle, angles.right_ankle,
            angles.left_back, angles.right_back, angles.knee_over_toe,
            angles.torso_lean, angles.neck_angle,
            angles.knee_width_ratio, angles.hip_level_diff,
            angles.shoulder_level, self.rep_count,
            "|".join([e.name for e in self._live_errors]) if self._live_errors else "none",
            self._live_score,
        ]
        write_header = not os.path.exists(self._dataset_file)
        with open(self._dataset_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(DATASET_HEADERS)
            writer.writerow(row)

    def toggle_dataset_recording(self):
        self._recording = not self._recording
        if self._recording:
            if not os.path.exists(self._dataset_file):
                with open(self._dataset_file, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(DATASET_HEADERS)
        print(f"[DATASET] Recording {'started' if self._recording else 'stopped'} to {self._dataset_file}")
    # Public toggles
    _show_stats = False

    def toggle_stats(self):
        self._show_stats = not self._show_stats


# ─────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────────────────────────

def main():
    print("━" * 60)
    print("  AI SQUAT TRAINER  –  Starting camera...")
    print("  Controls: Q/ESC=Quit | R=Reset | C=Calibrate | S=Stats | D=Data")
    print("━" * 60)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try camera index 1 (external webcam)
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check connection and permissions.")
        return

    # Set camera resolution and capture settings for faster performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    analyser = SquatAnalyser()

    window_name = "AI Squat Trainer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("[OK] Camera opened. Press C to calibrate standing pose first!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed – retrying...")
            time.sleep(0.05)
            continue

        # Flip horizontally (mirror view)
        frame = cv2.flip(frame, 1)

        # Process frame
        output = analyser.process(frame)

        cv2.imshow(window_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):  # Q or ESC
            break
        elif key in (ord('r'), ord('R')):
            analyser.rep_count = 0
            analyser.rep_history.clear()
            analyser.phase = SquatPhase.STANDING
            print("[RESET] Rep counter cleared.")
        elif key in (ord('c'), ord('C')):
            # Calibrate – re-process current frame to get landmarks
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = analyser.pose.process(rgb)
            if results.pose_landmarks:
                raw_lm = results.pose_landmarks
                cal_lm = raw_lm.landmark if hasattr(raw_lm, 'landmark') else raw_lm
                analyser.calibrate(cal_lm)
                print("[CAL] Calibration saved from standing pose.")
            else:
                print("[CAL] No person detected for calibration.")
        elif key in (ord('s'), ord('S')):
            analyser.toggle_stats()

    cap.release()
    cv2.destroyAllWindows()
    analyser.pose.close()

    # ── Session summary ─────────────────────────────────────────
    if analyser.rep_count > 0:
        scores = [r.score for r in analyser.rep_history]
        print("\n" + "━" * 60)
        print(f"  SESSION SUMMARY  –  {analyser.rep_count} reps")
        print(f"  Avg Score : {np.mean(scores):.0f}/100")
        print(f"  Best Rep  : {max(scores)}/100")
        print(f"  Worst Rep : {min(scores)}/100")
        for i, rep in enumerate(analyser.rep_history, 1):
            errs = ", ".join(e.value for e in rep.errors) or "No errors"
            print(f"  Rep {i:02d}: {rep.score}/100  depth={rep.depth_reached:.0f}°  {errs}")
        print("━" * 60)
    else:
        print("\nNo reps recorded this session.")


if __name__ == "__main__":
    main()
