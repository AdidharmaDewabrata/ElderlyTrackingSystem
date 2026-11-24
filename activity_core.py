"""activity_core.py - shared by the training notebook AND monitor_server.py.
Same code for training and inference, so features can never drift apart.

Key differences from the old pipeline:
  * features use REAL pixel coordinates (aspect-correct), never stretched crops
  * view-robust 3D joint angles from MediaPipe world landmarks
  * torso / thigh angle from vertical (the main signal for lying vs upright)
  * everything normalised by body extent, so camera distance doesn't matter
  * no brightness/blur preprocessing (it degrades landmark quality)
"""
import os
import urllib.request
import numpy as np
import cv2

CLASSES = ["falling", "sitting", "standing", "walking"]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
MODEL_PATH = "pose_landmarker_heavy.task"

NOSE, LS, RS, LE, RE, LW, RW, LH, RH, LK, RK, LA, RA = 0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28

FEATURE_NAMES = [
    "knee_l", "knee_r", "hip_l", "hip_r", "elbow_l", "elbow_r", "knee_mean", "hip_mean",
    "torso_angle", "thigh_angle", "shin_angle", "body_axis_angle", "log_aspect",
    "head_hip_dy", "shoulder_hip_dy", "hip_knee_dy", "knee_ankle_dy", "hip_knee_dx",
    "ankle_spread_x", "ankle_spread_y", "knee_spread_x", "torso_len", "vis_mean",
]


def ensure_pose_model(path=MODEL_PATH):
    if not os.path.exists(path):
        print("Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _angle3(a, b, c):
    """Joint angle at b in degrees, from 3D world landmarks (view-robust)."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def signed_from_vertical(v):
    """Angle of image vector v=(dx,dy) from 'straight up', degrees. 0=up, +-90=horizontal."""
    return float(np.degrees(np.arctan2(v[0], -v[1])))


def extract_features(xy, vis, w3):
    """xy: (33,2) pixel coords | vis: (33,) | w3: (33,3) world coords -> list of floats."""
    mid = lambda a, b: (xy[a] + xy[b]) / 2.0
    sh, hip, kn, an = mid(LS, RS), mid(LH, RH), mid(LK, RK), mid(LA, RA)
    pts = xy[vis > 0.3] if (vis > 0.3).sum() >= 6 else xy
    bw, bh = np.ptp(pts[:, 0]) + 1e-6, np.ptp(pts[:, 1]) + 1e-6
    ext = max(bw, bh)
    n = lambda v: float(v) / ext
    f = {
        "knee_l": _angle3(w3[LH], w3[LK], w3[LA]), "knee_r": _angle3(w3[RH], w3[RK], w3[RA]),
        "hip_l": _angle3(w3[LS], w3[LH], w3[LK]), "hip_r": _angle3(w3[RS], w3[RH], w3[RK]),
        "elbow_l": _angle3(w3[LS], w3[LE], w3[LW]), "elbow_r": _angle3(w3[RS], w3[RE], w3[RW]),
        "torso_angle": abs(signed_from_vertical(sh - hip)),
        "thigh_angle": abs(signed_from_vertical(hip - kn)),
        "shin_angle": abs(signed_from_vertical(kn - an)),
        "body_axis_angle": abs(signed_from_vertical(sh - an)),
        "log_aspect": float(np.log(bw / bh)),
        "head_hip_dy": n(xy[NOSE][1] - hip[1]), "shoulder_hip_dy": n(sh[1] - hip[1]),
        "hip_knee_dy": n(kn[1] - hip[1]), "knee_ankle_dy": n(an[1] - kn[1]),
        "hip_knee_dx": n(abs(kn[0] - hip[0])),
        "ankle_spread_x": n(abs(xy[LA][0] - xy[RA][0])), "ankle_spread_y": n(abs(xy[LA][1] - xy[RA][1])),
        "knee_spread_x": n(abs(xy[LK][0] - xy[RK][0])),
        "torso_len": n(np.linalg.norm(sh - hip)),
        "vis_mean": float(vis[[LS, RS, LH, RH, LK, RK, LA, RA]].mean()),
    }
    f["knee_mean"] = (f["knee_l"] + f["knee_r"]) / 2
    f["hip_mean"] = (f["hip_l"] + f["hip_r"]) / 2
    return [f[k] for k in FEATURE_NAMES]


class PoseEngine:
    """YOLO person crop (optional, never stretched) -> MediaPipe PoseLandmarker -> features."""

    def __init__(self, use_yolo=True, yolo_weights="yolov8n.pt"):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_pose_model()),
            running_mode=vision.RunningMode.IMAGE, num_poses=1,
            min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
        self.pose = vision.PoseLandmarker.create_from_options(opts)
        self.yolo = None
        if use_yolo:
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(yolo_weights)
            except Exception as e:  # YOLO is optional
                print("YOLO disabled:", e)

    def _person_box(self, bgr):
        r = self.yolo(bgr, classes=[0], conf=0.25, verbose=False)[0]
        if len(r.boxes) == 0:
            return None
        b = r.boxes.xyxy.cpu().numpy()
        return b[((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])).argmax()]

    def process(self, bgr):
        """Returns dict(features, keypoints[33,3] normalised to FULL frame, xy[33,2] crop px) or None."""
        import mediapipe as mp
        H, W = bgr.shape[:2]
        crop, ox, oy = bgr, 0, 0
        box = self._person_box(bgr) if self.yolo is not None else None
        if box is not None:
            x1, y1, x2, y2 = box
            px, py = 0.1 * (x2 - x1), 0.1 * (y2 - y1)
            x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
            x2, y2 = int(min(W, x2 + px)), int(min(H, y2 + py))
            if x2 - x1 > 32 and y2 - y1 > 32:
                crop, ox, oy = bgr[y1:y2, x1:x2], x1, y1
        ch, cw = crop.shape[:2]
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        res = self.pose.detect(img)
        if not res.pose_landmarks:
            return None
        lm = res.pose_landmarks[0]
        xy = np.array([[p.x * cw, p.y * ch] for p in lm])
        vis = np.array([p.visibility for p in lm])
        w3 = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]])
        full = (xy + [ox, oy]) / [W, H]
        return dict(features=extract_features(xy, vis, w3), keypoints=np.c_[full, vis], xy=xy)


def imread_safe(path):
    from PIL import Image
    try:
        return cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def video_frames(path, n=12):
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in np.linspace(0, max(total - 1, 0), n, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            out.append(fr)
    cap.release()
    return out


def extract_video_frames(video_path, frame_step=10, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()
    return frames


def augment(bgr):
    """Light, label-safe augmentation (flip + lighting). No rotation: it would corrupt torso angles."""
    return [cv2.flip(bgr, 1),
            cv2.convertScaleAbs(bgr, alpha=0.75, beta=-15),
            cv2.convertScaleAbs(bgr, alpha=1.25, beta=25)]
