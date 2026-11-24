import argparse
import os
import pickle
import sys
import warnings
from collections import Counter
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import cv2
import joblib
import numpy as np
import pandas as pd
import matplotlib

if "--show" not in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from activity_core import (
    CLASSES,
    FEATURE_NAMES,
    PoseEngine,
    extract_video_frames,
    augment,
)

warnings.filterwarnings("ignore")

ACTIVITIES = CLASSES
SHOW = False

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class Paths:
    def __init__(self, base):
        self.base = Path(base)
        self.dataset_raw = self.base / "dataset" / "raw"
        self.dataset_proc = self.base / "dataset" / "processed"
        self.models_dir = self.base / "models"
        self.results_dir = self.base / "results"
        self.features_csv = self.dataset_proc / "features.csv"


def get_video_files(folder):
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    return sorted([
        str(p) for p in folder_path.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    ])


def setup_folders(paths):
    for activity in ACTIVITIES:
        os.makedirs(paths.dataset_raw / activity, exist_ok=True)
    os.makedirs(paths.dataset_proc, exist_ok=True)
    os.makedirs(paths.models_dir, exist_ok=True)
    os.makedirs(paths.results_dir, exist_ok=True)

    print("Folder structure ready!")
    print(f"Base directory: {paths.base}")
    print("\nTaruh video (.mp4 / .avi) ke folder:")
    for a in ACTIVITIES:
        count = len(get_video_files(paths.dataset_raw / a))
        print(f"   {paths.dataset_raw / a} -> {count} video")


def show_or_close():
    if SHOW:
        plt.show()
    else:
        plt.close("all")


def draw_coco_skeleton(bgr, keypoints, conf_thresh=0.25):
    img_draw = bgr.copy()
    if keypoints is None:
        return img_draw

    h, w = img_draw.shape[:2]
    pts = [(int(x * w), int(y * h), c) for x, y, c in keypoints]

    for i, j in COCO_SKELETON:
        x1, y1, c1 = pts[i]
        x2, y2, c2 = pts[j]
        if c1 >= conf_thresh and c2 >= conf_thresh and (x1 > 0 or y1 > 0) and (x2 > 0 or y2 > 0):
            cv2.line(img_draw, (x1, y1), (x2, y2), (122, 199, 167), 2)

    for x, y, c in pts:
        if c >= conf_thresh and (x > 0 or y > 0):
            cv2.circle(img_draw, (x, y), 4, (0, 255, 0), -1)

    return img_draw


def load_and_extract(dataset_path, activities, engine, use_augmentation=True,
                     frame_step=10, max_frames_per_video=None):
    all_rows = []
    all_labels = []
    skipped_videos = 0
    total_frames = 0
    no_pose = 0

    for activity in activities:
        folder = os.path.join(dataset_path, activity)
        videos = get_video_files(folder)
        if not videos:
            print(f"[{activity}] Tidak ada video (.mp4/.avi) ditemukan - skip")
            continue

        print(f"\n[{activity}] Memproses {len(videos)} video (frame_step={frame_step})...")
        for video_path in tqdm(videos, desc=activity):
            frames = extract_video_frames(
                video_path, frame_step=frame_step, max_frames=max_frames_per_video
            )
            if not frames:
                skipped_videos += 1
                continue

            total_frames += len(frames)
            for bgr in frames:
                variants = [bgr] + (augment(bgr) if use_augmentation else [])
                for frame_var in variants:
                    out = engine.process(frame_var)
                    if out is None:
                        no_pose += 1
                        continue
                    all_rows.append(out["features"])
                    all_labels.append(activity)

    df = pd.DataFrame(all_rows, columns=FEATURE_NAMES)
    if not df.empty:
        df["label"] = all_labels
        df = df.dropna()

    print("\nDataset video selesai diproses!")
    print(f"   Frame diekstrak : {total_frames}")
    print(f"   Total sampel    : {len(df)}")
    print(f"   Skipped videos  : {skipped_videos}")
    print(f"   No pose         : {no_pose}")
    if not df.empty:
        print("\nDistribusi per kelas:")
        print(df["label"].value_counts())

    return df


def plot_distribution(df, results_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = df["label"].value_counts()
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0", "#F44336"][:len(counts)]

    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_title("Distribusi Kelas Aktivitas (Frame Samples)", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Aktivitas")
    axes[0].set_ylabel("Jumlah Sampel Frame")
    axes[0].tick_params(axis="x", rotation=15)
    for i, (_, val) in enumerate(counts.items()):
        axes[0].text(i, val + 1, str(val), ha="center", fontweight="bold")

    axes[1].pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90, colors=colors)
    axes[1].set_title("Proporsi Dataset", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(results_dir / "dataset_distribution.png", dpi=150, bbox_inches="tight")
    show_or_close()


def train_models(df):
    X = df[FEATURE_NAMES]
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Classes          : {list(le.classes_)}")

    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"Random Forest Accuracy: {rf_acc*100:.2f}%")

    print("\nTraining SVM (RBF kernel)...")
    svm_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
    ])
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    print(f"SVM Accuracy: {svm_acc*100:.2f}%")

    best_is_rf = rf_acc >= svm_acc
    best = {
        "model": rf_model if best_is_rf else svm_model,
        "pred": rf_pred if best_is_rf else svm_pred,
        "name": "Random Forest" if best_is_rf else "SVM",
        "acc": max(rf_acc, svm_acc),
    }
    print(f"\nBest Model: {best['name']} ({best['acc']*100:.2f}%)")

    return dict(
        X=X, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        le=le, rf_model=rf_model, svm_model=svm_model,
        rf_pred=rf_pred, svm_pred=svm_pred, rf_acc=rf_acc, svm_acc=svm_acc, best=best,
    )


def evaluate(t, results_dir):
    le, y_test = t["le"], t["y_test"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (name, pred) in zip(axes, [("Random Forest", t["rf_pred"]), ("SVM", t["svm_pred"])]):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_, ax=ax, linewidths=0.5)
        acc = accuracy_score(y_test, pred)
        ax.set_title(f"{name}\nAccuracy: {acc*100:.2f}%", fontsize=13, fontweight="bold")
        ax.set_ylabel("True Label")
        ax.set_xlabel("Predicted Label")
        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Confusion Matrix - Elderly Activity Classification", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    show_or_close()

    print("=" * 55)
    print(f"  CLASSIFICATION REPORT - {t['best']['name']}")
    print("=" * 55)
    print(classification_report(y_test, t["best"]["pred"], target_names=le.classes_))


def plot_feature_importance(t, results_dir):
    importances = t["rf_model"].feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.barh([FEATURE_NAMES[i] for i in indices[::-1]],
             importances[indices[::-1]], color="#2196F3", edgecolor="white")
    plt.xlabel("Feature Importance Score")
    plt.title("Top 15 Fitur Terpenting - Random Forest (YOLOv8-Pose)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(results_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    show_or_close()


def save_models(t, models_dir, base_dir):
    for name, obj in [("random_forest.pkl", t["rf_model"]),
                      ("svm.pkl", t["svm_model"]),
                      ("label_encoder.pkl", t["le"])]:
        with open(models_dir / name, "wb") as fh:
            pickle.dump(obj, fh)

    bundle = {
        "model": t["best"]["model"],
        "model_name": t["best"]["name"],
        "accuracy": float(t["best"]["acc"]),
        "classes": list(t["le"].classes_),
        "features": FEATURE_NAMES,
    }
    bundle_in_models = models_dir / "elderly_activity_model.pkl"
    bundle_in_root = Path(base_dir) / "elderly_activity_model.pkl"
    joblib.dump(bundle, bundle_in_models)
    joblib.dump(bundle, bundle_in_root)

    print("Models saved:")
    print(f"   {bundle_in_models}")
    print(f"   {bundle_in_root}")
    print(f"   {models_dir / 'random_forest.pkl'}")
    print(f"   {models_dir / 'svm.pkl'}")
    print(f"   {models_dir / 'label_encoder.pkl'}")


def show_sample_predictions(dataset_path, activities, engine, model, le, results_dir, n_per_class=2):
    n_activities = len(activities)
    fig, axes = plt.subplots(n_activities, n_per_class,
                             figsize=(n_per_class * 4, n_activities * 4), squeeze=False)

    for row, activity in enumerate(activities):
        videos = get_video_files(os.path.join(dataset_path, activity))
        sample_frames = []
        if len(videos) >= n_per_class:
            for v_path in videos[:n_per_class]:
                v_frames = extract_video_frames(v_path, frame_step=15, max_frames=2)
                if v_frames:
                    sample_frames.append(v_frames[len(v_frames) // 2])
        elif len(videos) == 1:
            v_frames = extract_video_frames(videos[0], frame_step=15, max_frames=n_per_class)
            if v_frames:
                sample_frames.extend(v_frames[:n_per_class])

        for col in range(n_per_class):
            ax = axes[row][col]
            if col >= len(sample_frames):
                ax.axis("off")
                continue

            bgr = sample_frames[col]
            out = engine.process(bgr)
            rgb_draw = cv2.cvtColor(
                draw_coco_skeleton(bgr, out["keypoints"] if out else None),
                cv2.COLOR_BGR2RGB,
            )

            if out is not None:
                X_one = pd.DataFrame([out["features"]], columns=FEATURE_NAMES)
                pred_enc = model.predict(X_one)[0]
                pred_lbl = le.inverse_transform([pred_enc])[0]
                conf = model.predict_proba(X_one).max() * 100
                color = "green" if pred_lbl == activity else "red"
                ax.set_title(f"True: {activity}\nPred: {pred_lbl} ({conf:.0f}%)",
                             color=color, fontsize=9, fontweight="bold")
            else:
                ax.set_title(f"True: {activity}\nNo pose detected", color="gray", fontsize=9)

            ax.imshow(rgb_draw)
            ax.axis("off")

    plt.suptitle("Sample Predictions - Elderly Activity Monitoring (YOLOv8-Pose)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(results_dir / "sample_predictions.png", dpi=150, bbox_inches="tight")
    show_or_close()


def predict_video(video_path, engine, model, classes, frame_step=10, save_path=None):
    frames = extract_video_frames(video_path, frame_step=frame_step)
    if not frames:
        print("Video tidak bisa dibaca atau kosong")
        return

    predictions, confidences = [], []
    last_bgr, last_kpts = None, None

    for bgr in frames:
        out = engine.process(bgr)
        if out is None:
            continue
        X_one = pd.DataFrame([out["features"]], columns=FEATURE_NAMES)
        probs = model.predict_proba(X_one)[0]
        idx = int(np.argmax(probs))
        predictions.append(classes[idx])
        confidences.append(float(probs[idx]) * 100)
        last_bgr, last_kpts = bgr, out["keypoints"]

    if not predictions:
        print("Pose tidak terdeteksi di seluruh frame video")
        return

    majority_label, votes = Counter(predictions).most_common(1)[0]
    avg_conf = np.mean([c for p, c in zip(predictions, confidences) if p == majority_label])

    print(f"\nPredicted Activity : {majority_label.upper()}")
    print(f"   Frame Votes     : {votes}/{len(predictions)} frame")
    print(f"   Avg Confidence  : {avg_conf:.1f}%")

    if save_path and last_bgr is not None:
        vis = draw_coco_skeleton(last_bgr, last_kpts)
        cv2.putText(vis, f"{majority_label.upper()} ({avg_conf:.0f}%)", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(str(save_path), vis)
        print(f"Preview saved to: {save_path}")


def run_train(args, paths):
    setup_folders(paths)
    engine = PoseEngine()

    if args.reuse_features and paths.features_csv.exists():
        print(f"\nMemakai features yang sudah ada: {paths.features_csv}")
        df = pd.read_csv(paths.features_csv)
    else:
        df = load_and_extract(
            dataset_path=str(paths.dataset_raw),
            activities=ACTIVITIES,
            engine=engine,
            use_augmentation=not args.no_augment,
            frame_step=args.frame_step,
            max_frames_per_video=args.max_frames,
        )
        if df.empty or "label" not in df.columns:
            raise SystemExit(f"\nTidak ada sampel yang berhasil diproses.\n"
                             f"Taruh video .mp4/.avi di {paths.dataset_raw}/<aktivitas>/ lalu jalankan lagi.")
        df.to_csv(paths.features_csv, index=False)
        print(f"\nFeatures saved to: {paths.features_csv}")

    plot_distribution(df, paths.results_dir)
    t = train_models(df)
    evaluate(t, paths.results_dir)
    plot_feature_importance(t, paths.results_dir)
    save_models(t, paths.models_dir, paths.base)
    show_sample_predictions(str(paths.dataset_raw), ACTIVITIES, engine,
                            t["best"]["model"], t["le"], paths.results_dir, n_per_class=2)
    print("\nSelesai! Jalankan dashboard dengan: python monitor_server.py")


def run_predict(args, paths):
    target_video = args.video or args.image
    if not target_video:
        raise SystemExit("Pakai: python elderly_activity_monitoring.py predict --video video.mp4")

    bundle_path = paths.base / "elderly_activity_model.pkl"
    if not bundle_path.exists():
        bundle_path = paths.models_dir / "elderly_activity_model.pkl"
    if not bundle_path.exists():
        raise SystemExit("Model belum ada. Jalankan dulu: python elderly_activity_monitoring.py train")

    bundle = joblib.load(bundle_path)
    engine = PoseEngine()
    os.makedirs(paths.results_dir, exist_ok=True)
    predict_video(
        target_video, engine, bundle["model"], bundle["classes"],
        frame_step=args.frame_step, save_path=paths.results_dir / "prediction.png"
    )


def main():
    global SHOW
    ap = argparse.ArgumentParser(description="Elderly Smart Activity Monitoring (YOLOv8-Pose + .mp4/.avi)")
    ap.add_argument("command", nargs="?", default="train", choices=["train", "predict"])
    ap.add_argument("--base", default=str(Path(__file__).resolve().parent))
    ap.add_argument("--frame-step", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--reuse-features", action="store_true")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--video", help="[predict] path video .mp4/.avi")
    ap.add_argument("--image", help="[predict] alias untuk --video")
    args = ap.parse_args()

    SHOW = args.show
    paths = Paths(args.base)

    if args.command == "predict":
        run_predict(args, paths)
    else:
        run_train(args, paths)


if __name__ == "__main__":
    main()