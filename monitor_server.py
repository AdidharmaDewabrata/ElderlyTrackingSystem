"""monitor_server.py - run:  python monitor_server.py   then open http://localhost:5000
Needs: activity_core.py, elderly_activity_model.pkl (from the notebook), dashboard.html in the same folder.
pip install flask joblib pandas scikit-learn mediapipe opencv-python ultralytics
"""
import os, time, threading
from collections import deque
import numpy as np, pandas as pd, cv2, joblib
from flask import Flask, request, jsonify, send_from_directory
from activity_core import PoseEngine, FEATURE_NAMES, signed_from_vertical

bundle = joblib.load(os.environ.get("MODEL", "elderly_activity_model.pkl"))
CLS = list(bundle["classes"])
engine = PoseEngine(use_yolo=os.environ.get("YOLO", "1") == "1")
lock = threading.Lock()
app = Flask(__name__)


class Tracker:
    """Turns noisy per-frame probabilities into a stable label + fall alert.
    Fall  = sudden downward drop / upright->horizontal, then lying (or model says 'falling' for 2s).
    Sleep = lying WITHOUT a preceding drop. Thresholds are tunable below."""
    DROP_RATIO, UPRIGHT_DEG, LYING_DEG, WALK_V = 0.30, 35, 50, 0.30

    def __init__(s):
        s.reset()

    def reset(s):
        s.hist = deque(maxlen=120); s.ema = None; s.label = "standing"; s.cand = None; s.cand_n = 0
        s.fall = False; s.drop_t = -99.0; s.last_drop_ev = -99.0; s.vote_t = None; s.up_t = None; s.events = []

    def ev(s, text, lvl="ok"):
        s.events.append(dict(text=text, lvl=lvl))

    def update(s, probs, torso, hip_y, ext, cx, now):
        p = np.array([probs.get(c, 0.0) for c in CLS])
        s.ema = p if s.ema is None else 0.6 * s.ema + 0.4 * p
        P = dict(zip(CLS, s.ema))
        lying_mass = P.get("falling", 0)
        s.hist.append((now, torso, hip_y, ext, cx))
        old = next(h for h in s.hist if now - h[0] <= 1.2)

        # --- sudden drop detection (1.2 s window) ---
        dropped = (hip_y - old[2]) / max(old[3], 1) > s.DROP_RATIO or \
                  (old[1] < s.UPRIGHT_DEG and torso > 60 and now - old[0] > 0.2)
        if dropped:
            s.drop_t = now
            if now - s.last_drop_ev > 3:
                s.ev("Sudden Acceleration / Downward Drop", "crit"); s.last_drop_ev = now

        lying = torso > s.LYING_DEG and (lying_mass > 0.45 or torso > 60)
        new = s.label
        if s.fall:
            new = "falling"
            if torso < s.UPRIGHT_DEG and (lying_mass < 0.3 or torso < 35):
                s.up_t = s.up_t or now
                if now - s.up_t > 4:
                    s.fall = False; s.ev("Resident upright again - fall cleared"); new = "standing"
            else:
                s.up_t = None
        elif lying:
            if now - s.drop_t < 6:
                trigger = True
            elif P.get("falling", 0) > 0.5:
                s.vote_t = s.vote_t or now; trigger = now - s.vote_t >= 2
            else:
                s.vote_t = None; trigger = False
            if trigger:
                s.fall = True; s.ev("Posture shift: Upright to Lateral", "crit"); new = "falling"
            else:
                new = s.label
        else:
            s.vote_t = None
            up = {k: P.get(k, 0) for k in ("sitting", "standing", "walking")}
            new = max(up, key=up.get)
            span = now - s.hist[0][0]
            v = abs(cx - next(h for h in s.hist if now - h[0] <= 1.0)[4]) / max(ext, 1)
            if v > s.WALK_V and new != "sitting": new = "walking"
            elif new == "walking" and v < 0.05 and span > 1.5: new = "standing"

        # hysteresis: 3 consecutive frames before changing (a fall is applied immediately)
        if new != s.label:
            s.cand_n = s.cand_n + 1 if new == s.cand else 1; s.cand = new
            if new == "falling" or s.cand_n >= 3:
                if new != "falling":
                    s.ev("Ambulatory walking (gait normal)" if new == "walking" else f"Activity: {new.capitalize()}")
                s.label, s.cand_n = new, 0
        else:
            s.cand_n = 0

        grp = (lying_mass if lying_mass > 0 else 0.95) if s.label == "falling" else \
            (P.get("sitting", 0) if s.label == "sitting" else P.get("standing", 0) + P.get("walking", 0))
        return s.label, float(min(grp, 0.999))


T = Tracker()


@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"] = "*"
    r.headers["Access-Control-Allow-Headers"] = "*"
    return r


@app.get("/")
def index():
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")


@app.get("/health")
def health():
    return jsonify(ok=True, classes=CLS, yolo=engine.yolo is not None)


@app.post("/ack")
def ack():
    with lock:
        T.fall = False; T.drop_t = -99.0; T.vote_t = None; T.label = "standing"; T.ev("Alert acknowledged by staff")
    return jsonify(ok=True)


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)
    t0 = time.time()
    data = request.files["frame"].read() if "frame" in request.files else request.get_data()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify(ok=False, error="bad image"), 400
    H, W = img.shape[:2]
    with lock:
        out = engine.process(img)
        if out is None:
            ev, T.events = T.events, []
            return jsonify(ok=True, person=False, label=T.label, fall_active=T.fall, events=ev,
                           latency_ms=round((time.time() - t0) * 1000))
        X = pd.DataFrame([out["features"]], columns=FEATURE_NAMES)
        probs = dict(zip(CLS, bundle["model"].predict_proba(X)[0]))
        kp = out["keypoints"]
        fp = kp[:, :2] * [W, H]                                  # full-frame pixels
        sh, hip = (fp[11] + fp[12]) / 2, (fp[23] + fp[24]) / 2
        torso = abs(signed_from_vertical(sh - hip))
        ext = max(np.ptp(fp[:, 0]), np.ptp(fp[:, 1]))
        label, conf = T.update(probs, torso, float(hip[1]), float(ext), float(hip[0]), time.time())
        ev, T.events = T.events, []
    return jsonify(ok=True, person=True, label=label, confidence=conf,
                   probs={k: round(float(v), 3) for k, v in probs.items()},
                   keypoints=np.round(kp, 4).tolist(), fall_active=T.fall, events=ev,
                   latency_ms=round((time.time() - t0) * 1000))


if __name__ == "__main__":
    print("Dashboard -> http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, threaded=False)
