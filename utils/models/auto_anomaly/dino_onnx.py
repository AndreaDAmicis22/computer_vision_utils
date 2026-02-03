import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

# from packages.redis_utils import redis_utils


# -------------------------------------------------
# Math utilities (NumPy)
# -------------------------------------------------
def compute_mean_cov_inv(features, eps=1e-2):
    """
    features: (N, D)
    """
    mean = np.mean(features, axis=0)
    diffs = features - mean
    cov = diffs.T @ diffs / (features.shape[0] - 1)
    cov += eps * np.eye(cov.shape[0], dtype=np.float32)
    cov_inv = np.linalg.inv(cov)
    return mean, cov_inv


def mahalanobis_distance(vec, mean, cov_inv):
    """
    vec: (N, D)
    """
    diff = vec - mean
    d_sq = np.sum((diff @ cov_inv) * diff, axis=1)
    return np.sqrt(d_sq)


# -------------------------------------------------
# Sliding Window Anomaly Detector (ONNX)
# -------------------------------------------------
class SlidingWindowAnomalyDetectorONNX:
    def __init__(
        self,
        onnx_model_path,
        warmup_window_size=15,
        inference_window_size=30,
        ramp_start=16,
        ramp_end=50,
        start_gamma=0.3,
        area_threshold=200,
        global_area_threshold=200,
        anomaly_min=100.0,
        anomaly_max=1300.0,
        target_img_size=512,
        anomaly_threshold=0.5,
        providers=("OpenVINOExecutionProvider", "CPUExecutionProvider"),
    ):
        self.warmup_window_size = warmup_window_size
        self.inference_window_size = inference_window_size
        self.ramp_start = ramp_start
        self.ramp_end = ramp_end
        self.start_gamma = start_gamma
        self.area_threshold = area_threshold
        self.global_area_threshold = global_area_threshold
        self.anomaly_min = anomaly_min
        self.anomaly_max = anomaly_max
        self.anomaly_threshold = anomaly_threshold
        self.target_img_size = target_img_size

        # ONNX Runtime session

        providers = [
            ("OpenVINOExecutionProvider", {"device_type": "GPU", "precision": "FP16"}),
            ("CPUExecutionProvider", {}),
        ]
        if onnx_model_path is not None:
            self.session = ort.InferenceSession(
                onnx_model_path,
                providers=list(providers),
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            assert len(self.output_names) == 2, "ONNX model must output (cls, patches)"

        # Sliding window memory
        self.memory_cls_features = deque(maxlen=inference_window_size)
        self.memory_patch_features = deque(maxlen=inference_window_size)
        self.good_cls_distances = deque(maxlen=inference_window_size)
        self.good_patch_scores = deque(maxlen=inference_window_size)

        self.global_idx = 0
        self.warmup_done = False
        self._warmup_state_exists: bool = False

    # -------------------------------------------------
    # Preprocessing
    # -------------------------------------------------
    def _preprocess(self, img: np.ndarray):
        if not isinstance(img, np.ndarray):
            img = np.array(img)
        img = cv2.resize(img, (self.target_img_size, self.target_img_size), cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # normalize
        img = np.transpose(img, (2, 0, 1))  # CHW
        return img[np.newaxis, ...]  # (1, 3, H, W)

    # -------------------------------------------------
    # Feature extraction via ONNX
    # -------------------------------------------------
    def _extract_features(self, x):
        cls, patches = self.session.run(
            self.output_names,
            {self.input_name: x},
        )
        return patches[0], cls[0]  # remove batch dim

    def _save_warmup(self, path: Path):
        # disc_id = redis_utils.get_key_val("codice_prodotto", "int")
        # path = Path("/workspace/src/SPA006/state.npz")
        patch_stack = np.concatenate(self.memory_patch_features, axis=0)

        np.savez_compressed(
            path,
            memory_cls_features=np.stack(self.memory_cls_features),
            patch_stack=patch_stack,
            good_cls_distances=np.array(self.good_cls_distances),
            good_patch_scores=np.array(self.good_patch_scores),
            threshold=self.threshold,
        )

    def load_warmup(self, path):
        data = np.load(path)

        self.memory_cls_features = deque(
            data["memory_cls_features"],
            maxlen=self.inference_window_size,
        )

        # Rebuild patch memory as a single element (works perfectly)
        self.memory_patch_features = deque(
            [data["patch_stack"]],
            maxlen=self.inference_window_size,
        )

        self.good_cls_distances = deque(
            data["good_cls_distances"],
            maxlen=self.inference_window_size,
        )
        self.good_patch_scores = deque(
            data["good_patch_scores"],
            maxlen=self.inference_window_size,
        )

        self.threshold = float(data["threshold"])
        self.warmup_done = True
        self._warmup_state_exists = True

    def _create_anomaly_image_threshold(
        self,
        idx,
        img_pil,
        anomaly_map_norm,
        threshold=0.6,
        color=(255, 100, 0),
    ):
        """
        Overlay only anomaly regions above `threshold`.

        color: BGR tuple for overlay (default red)
        """
        img_np = np.array(img_pil).astype(np.uint8)

        if anomaly_map_norm is None:
            return img_np, None

        # Resize anomaly map to match image
        anomaly_map_resized = cv2.resize(
            anomaly_map_norm,
            (img_np.shape[1], img_np.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        # Create mask of regions above threshold
        mask = anomaly_map_resized > threshold

        # Prepare overlay image (same size as original)
        overlay = np.zeros_like(img_np, dtype=np.uint8)
        overlay[mask] = color  # paint only masked pixels
        mask = overlay.copy()
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)[1]

        # Blend overlay with original image
        alpha = 0.2
        overlayed = cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0)
        return overlayed, mask

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    def fit(self, images_path: str, save_path: str):
        """
        Esegue il warmup/training estraendo feature dalle immagini e salvando lo stato.
        """
        images_path = Path(images_path)
        save_path = Path(save_path)

        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = [f for f in images_path.iterdir() if f.suffix.lower() in valid_extensions]

        if not image_files:
            print(f"NOT FOUND IMAGES in {images_path}")
            return

        for img_path in tqdm(image_files):
            try:
                image_bgr = cv2.imread(str(img_path))

                if image_bgr is None:
                    print(f"Impossibile leggere: {img_path}")
                    continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                x = self._preprocess(image_rgb)
                patch_tokens, cls_token = self._extract_features(x)
                if len(self.memory_cls_features) >= 2:
                    # CLS distance
                    cls_stack = np.stack(self.memory_cls_features)

                    mean_cls, covinv_cls = compute_mean_cov_inv(cls_stack)
                    dist_cls = mahalanobis_distance(cls_token[None, :], mean_cls, covinv_cls)[0]
                    # Patch anomaly map
                    patch_stack = np.concatenate(self.memory_patch_features, axis=0)
                    mean_patch, covinv_patch = compute_mean_cov_inv(patch_stack)
                    num_real_patches = (self.target_img_size // 16) ** 2
                    patch_tokens = patch_tokens[-num_real_patches:]
                    diff = patch_tokens - mean_patch
                    patch_scores = np.einsum("nd,df,nf->n", diff, covinv_patch, diff)

                    self.good_patch_scores.extend(patch_scores)
                else:
                    dist_cls = 0.0

                self.memory_cls_features.append(cls_token)
                self.memory_patch_features.append(patch_tokens)
                self.good_cls_distances.append(dist_cls)

            except Exception as e:
                print(f"Errore durante l'elaborazione di {img_path}: {e}")
                continue

        std_cls = np.std(self.good_cls_distances)
        p99 = np.percentile(self.good_cls_distances, 99)
        self.threshold = 1.0 * p99 + std_cls

        self._save_warmup(path=save_path)
        self.warmup_done = True
        print(f"Warmup completato con {len(self.memory_cls_features)} immagini.")
        print(f"Stato salvato in: {save_path}")

    # -------------------------------------------------
    # Single image processing
    # -------------------------------------------------
    def _process_single_image(self, image: np.ndarray):
        self.global_idx += 1
        idx = self.global_idx

        x = self._preprocess(image)
        time.time()
        patch_tokens, cls_token = self._extract_features(x)

        if len(self.memory_cls_features) >= 2:
            # CLS distance
            cls_stack = np.stack(self.memory_cls_features)

            mean_cls, covinv_cls = compute_mean_cov_inv(cls_stack)
            dist_cls = mahalanobis_distance(cls_token[None, :], mean_cls, covinv_cls)[0]
            # Patch anomaly map
            patch_stack = np.concatenate(self.memory_patch_features, axis=0)
            mean_patch, covinv_patch = compute_mean_cov_inv(patch_stack)
            num_real_patches = (self.target_img_size // 16) ** 2
            patch_tokens = patch_tokens[-num_real_patches:]
            diff = patch_tokens - mean_patch
            patch_scores = np.einsum("nd,df,nf->n", diff, covinv_patch, diff)

            if not self.warmup_done:
                self.good_patch_scores.extend(patch_scores)
            grid = int(np.sqrt(num_real_patches))
            anomaly_map = patch_scores.reshape(grid, grid)
            p5 = np.percentile(self.good_patch_scores, 5)
            p95 = np.percentile(self.good_patch_scores, 95)

            anomaly_map_norm = np.clip((anomaly_map - p5) / (p95 - p5 + 1e-6), 0.0, 1.0)
            anomaly_map_up = cv2.resize(
                anomaly_map_norm,
                (x.shape[3], x.shape[2]),
                interpolation=cv2.INTER_LINEAR,
            )

            binary_mask_np = (anomaly_map_up > self.anomaly_threshold).astype(np.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_np, 8)

            total_area = 0
            filtered_mask = np.zeros_like(binary_mask_np)
            for stat_idx, stat in enumerate(stats[1:], start=1):
                _, _, w, h, _area = stat
                # print("w*h: ", w * h)
                if w * h >= self.area_threshold:
                    filtered_mask[labels == stat_idx] = 1
                    total_area += w * h
        else:
            dist_cls = 0.0
            total_area = 0.0
            anomaly_map_norm = None
            filtered_mask = None

        # Dynamic threshold
        if len(self.good_cls_distances) > 1:
            if not self.warmup_done:
                p99 = np.percentile(self.good_cls_distances, 70)
                std_cls = np.std(self.good_cls_distances)
                if idx <= self.ramp_start:
                    gamma = self.start_gamma
                else:
                    gamma = self.start_gamma + (1.0 - self.start_gamma) * (
                        (idx - self.ramp_start) / (self.ramp_end - self.ramp_start)
                    )
                threshold = gamma * p99 + std_cls
                # threshold = gamma * p70
                self.threshold = threshold
            else:
                threshold = self.threshold
        else:
            threshold = np.inf

        is_anomaly = self.warmup_done and dist_cls > threshold and total_area > self.global_area_threshold

        # Update memory
        if not is_anomaly and not self.warmup_done:
            self.memory_cls_features.append(cls_token)
            self.memory_patch_features.append(patch_tokens)
            self.good_cls_distances.append(dist_cls)

        if not self.warmup_done and len(self.memory_cls_features) >= self.warmup_window_size:
            self.warmup_done = True
            if not self._warmup_state_exists:
                self._save_warmup(path="/workspace/src/SPA006/state_running.npz")

        return {
            "idx": idx,
            "is_anomaly": is_anomaly,
            "dist_cls": dist_cls,
            "image": image,
            "threshold": self.threshold,
            "anomaly_map_norm": anomaly_map_norm,
            "binary_mask": filtered_mask,
        }

    # -------------------------------------------------
    # Sliding window loop
    # -------------------------------------------------
    def run(self, image: np.ndarray):
        last_out = None

        out = self._process_single_image(image)
        last_out = out

        drawn_contours = np.zeros(image.shape[:2], np.uint8)

        print(  # noqa: T201
            f"Image {out['idx']} | "
            f"dist={out['dist_cls']:.3f} | "
            f"thresh cls {self.threshold} | "
            f"anomaly={out['is_anomaly']} | "
            f"warmup_done={self.warmup_done}"
        )

        if out["is_anomaly"]:
            overlayed, mask = self._create_anomaly_image_threshold(
                out["idx"],
                out["image"],
                out["anomaly_map_norm"],
                threshold=self.anomaly_threshold,
            )
            return overlayed, mask, True

        if last_out is None:
            return image, drawn_contours, False

        return image, drawn_contours, False
