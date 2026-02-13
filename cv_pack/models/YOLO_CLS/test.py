import os

import cv2
import numpy as np
from cv_pack.models.YOLO_CLS.yolo import YOLO_CLS_ONNX
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm


def load_dataset(root):
    class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    samples = []
    for cls_id, cls_name in enumerate(class_names):
        folder = os.path.join(root, cls_name)
        for f in os.listdir(folder):
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                samples.append((os.path.join(folder, f), cls_id))
    return samples, class_names


def evaluate(model, samples):
    y_true, y_pred = [], []

    for img_path, gt in tqdm(samples):
        img = cv2.imread(img_path)
        out = model.predict(img)
        y_pred.append(out["class_id"])
        y_true.append(gt)

    return np.array(y_true), np.array(y_pred)


if __name__ == "__main__":
    ROOT = "/mnt/nas/datasets/tnr004/ricciolo/20260123_cls_test"
    MODEL_PATH = "/workspace/src/TNR004/projects/tnr005/classification/2026-01-21/weights/best.onnx"

    model = YOLO_CLS_ONNX(MODEL_PATH, input_size=640, use_cuda=True)

    samples, class_names = load_dataset(ROOT)
    y_true, y_pred = evaluate(model, samples)

    print("\n📊 Classification report:\n")  # noqa: T201
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))  # noqa: T201

    print("\n🧮 Confusion matrix:\n")  # noqa: T201
    print(confusion_matrix(y_true, y_pred))  # noqa: T201
