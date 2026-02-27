# l5090_eval.py

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple, Optional
import traceback
import os


# =========================================================
# Model Factory
# =========================================================

class BaseDetector:
    def predict(self, frame: np.ndarray):
        raise NotImplementedError


class YOLOPtDetector(BaseDetector):
    def __init__(self, weight_path: str, imgsz: int, target_class: List[int]):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics 未安裝。請先執行: pip install ultralytics"
            )

        self.model = YOLO(weight_path)
        self.imgsz = imgsz
        self.target_class = target_class

    def predict(self, frame: np.ndarray):
        results = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            classes=self.target_class,
            verbose=False
        )
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return False
        return True


class TFLiteDetector(BaseDetector):
    def __init__(self, weight_path: str, imgsz: int):
        import tensorflow as tf

        self.imgsz = imgsz
        self.interpreter = tf.lite.Interpreter(model_path=weight_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def predict(self, frame: np.ndarray):
        frame = cv2.resize(frame, (self.imgsz, self.imgsz))
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)

        self.interpreter.set_tensor(
            self.input_details[0]["index"], frame
        )
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )

        return np.max(output) > 0.5


def _build_detector(weight_path: str, imgsz: int, target_class: List[int]) -> BaseDetector:
    suffix = Path(weight_path).suffix.lower()

    if suffix == ".pt":
        return YOLOPtDetector(weight_path, imgsz, target_class)
    elif suffix == ".tflite":
        return TFLiteDetector(weight_path, imgsz)
    else:
        raise ValueError(f"不支援的模型格式: {suffix}")


# =========================================================
# Core Evaluation Logic
# =========================================================

def _list_videos(root: Path):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]


def _rle_smooth(binary_arr: np.ndarray, k: int):
    out = np.zeros_like(binary_arr)
    n = len(binary_arr)
    i = 0
    while i < n:
        if binary_arr[i] == 1:
            j = i
            while j < n and binary_arr[j] == 1:
                j += 1
            if j - i >= k:
                out[i:j] = 1
            i = j
        else:
            i += 1
    return out


def _fit_logistic(x, y):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x.reshape(-1, 1), y)

    a = clf.intercept_[0]
    b = clf.coef_[0][0]

    def inv_logit(p):
        if b == 0:
            return None
        return (np.log(p / (1 - p)) - a) / b

    return inv_logit(0.5), inv_logit(0.9), clf


# =========================================================
# Public API
# =========================================================

def evaluate_L5090(
    save_dir: str,
    weight_path: str,
    target_class: List[int],
    imgsz: int,
    video_root: str = "/kaggle/input/pet-yolo-zoom-out-vid/zoom_out",
    k_consec_miss: int = 20,
    bin_width: int = 3,
):
    """
    回傳 (L50, L90)
    """

    try:
        detector = _build_detector(weight_path, imgsz, target_class)
    except Exception as e:
        print("模型初始化失敗：")
        print(e)
        print(traceback.format_exc())
        return 100, 100

    # 建立輸出資料夾
    model_name = Path(weight_path).stem
    tpe_time = datetime.now(ZoneInfo("Asia/Taipei")).strftime("%Y%m%d_%H%M%S")
    out_root = Path(save_dir) / f"{model_name}_{tpe_time}"
    out_root.mkdir(parents=True, exist_ok=True)

    videos = _list_videos(Path(video_root))
    if not videos:
        print("沒有找到影片")
        return 100, 100

    sizes = []
    misses = []

    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detected = detector.predict(frame)
            size_estimate = min(frame.shape[:2]) * 0.5  # 手動固定物件比例

            sizes.append(size_estimate)
            misses.append(0 if detected else 1)

        cap.release()

    sizes = np.array(sizes)
    misses = np.array(misses)

    miss_smooth = _rle_smooth(misses, k_consec_miss)

    L50, L90, clf = _fit_logistic(sizes, miss_smooth)

    # 繪圖
    xs = np.linspace(sizes.min(), sizes.max(), 300)
    probs = clf.predict_proba(xs.reshape(-1, 1))[:, 1]

    plt.figure(figsize=(8, 5))
    plt.scatter(sizes, miss_smooth, s=5, alpha=0.3)
    plt.plot(xs, probs)
    if L50:
        plt.axvline(L50)
    if L90:
        plt.axvline(L90)

    plt.xlabel("Short side size (px)")
    plt.ylabel("Miss probability")
    plt.tight_layout()
    plt.savefig(out_root / "L5090_curve.png")
    plt.close()

    pd.DataFrame({
        "size": sizes,
        "miss_smooth": miss_smooth
    }).to_csv(out_root / "per_frame.csv", index=False)

    print(f"完成。輸出位置: {out_root}")
    print(f"L50={L50}, L90={L90}")

    return L50, L90