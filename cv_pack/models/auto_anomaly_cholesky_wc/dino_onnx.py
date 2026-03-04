import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import openvino as ov
from matplotlib import pyplot as plt

# from scipy.linalg import solve_triangular

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s.%(msecs)03d | %(threadName)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("AnomalyDetector")


# -------------------------------------------------
# Math utilities (NumPy)
# -------------------------------------------------
def cholesky(features, eps=1e-2):
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=0)
    diffs = features - mean
    cov = diffs.T @ diffs
    cov /= features.shape[0] - 1
    cov.flat[:: cov.shape[0] + 1] += eps
    L = np.linalg.cholesky(cov)
    return mean, L


def mahalanobis_cholesky(diff, L):
    """
    Computes distance using the Cholesky factor L.
    diff: (N, D) - centered features
    L: (D, D) - Lower triangular matrix from cholesky(cov)
    """
    # Solve L*y = diff.T -> y is (D, N)
    # Using solve_triangular is faster than general np.linalg.solve
    y = np.linalg.solve(L, diff.T)
    return np.sum(y**2, axis=0)


# -------------------------------------------------
# Anomaly Detector (ONNX)
# -------------------------------------------------
class AnomalyDetectorONNX:
    def __init__(
        self,
        onnx_model_path,
        warmup_window_size=20,
        area_threshold=200,
        global_area_threshold=300,
        target_img_size=512,
        anomaly_threshold=0.6,
    ):
        self.warmup_window_size = warmup_window_size
        self.area_threshold = area_threshold
        self.global_area_threshold = global_area_threshold
        self.anomaly_threshold = anomaly_threshold
        self.target_img_size = target_img_size

        # ONNX Runtime session
        devices = ov.Core().available_devices
        if "GPU" in devices:
            providers = [
                ("OpenVINOExecutionProvider", {"device_type": "GPU", "precision": "FP16"}),
                ("CPUExecutionProvider", {}),
            ]
        else:
            providers = [
                ("OpenVINOExecutionProvider", {"device_type": "CPU", "precision": "FP32"}),
                ("CPUExecutionProvider", {}),
            ]

        logger.info(list(providers))
        if onnx_model_path is not None:
            self.session = ort.InferenceSession(
                onnx_model_path,
                providers=list(providers),
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            assert len(self.output_names) == 2, "ONNX model must output (cls, patches)"

        # Memory
        self.memory_cls_features = deque(maxlen=warmup_window_size)
        self.memory_patch_features = deque(maxlen=warmup_window_size)
        self.good_cls_distances = deque(maxlen=warmup_window_size)
        self.good_patch_scores = deque(maxlen=warmup_window_size)

        self.cached_mean_cls = None
        self.cached_L_cls = None
        self.cached_mean_patch = None
        self.cached_L_patch = None

        self.threshold = 0
        self.global_idx = 0
        self.warmup_done = False
        self._warmup_state_exists: bool = False

        self.target_img_size = target_img_size

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

    def _save_warmup(
        self, path: Path, memory_cls_features, patch_stack, good_cls_distances, good_patch_scores, threshold
    ):
        try:
            np.savez_compressed(
                path,
                memory_cls_features=memory_cls_features,
                patch_stack=patch_stack,
                good_cls_distances=good_cls_distances,
                good_patch_scores=good_patch_scores,
                threshold=threshold,
            )
            logger.info(f"Salvato stato in: {path} 📀")
        except Exception as e:
            logger.exception(f"Errore durante il salvataggio asincrono: {e}")

    def load_warmup(self, path):
        data = np.load(path)
        self.memory_cls_features = deque(
            data["memory_cls_features"],
            maxlen=self.warmup_window_size,
        )
        self.memory_patch_features = deque(
            [data["patch_stack"]],
            maxlen=self.warmup_window_size,
        )
        self.good_cls_distances = deque(
            data["good_cls_distances"],
            maxlen=self.warmup_window_size,
        )
        self.good_patch_scores = deque(
            data["good_patch_scores"],
            maxlen=self.warmup_window_size,
        )
        cls_stack = np.stack(list(self.memory_cls_features))
        self.cached_mean_cls, self.cached_L_cls = cholesky(cls_stack)
        patch_stack = self.memory_patch_features[0]
        self.cached_mean_patch, self.cached_L_patch = cholesky(patch_stack)
        self.threshold = float(data["threshold"])
        self.warmup_done = True
        self._warmup_state_exists = True
        logger.info(f"Caricato stato da: {path} ✅")

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

        return overlayed, mask, colored_map
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

        # Colored map
        if anomaly_map_resized.max() > 1.0:
            anomaly_map_resized = anomaly_map_resized / 255.0
        cm = plt.get_cmap("inferno")
        colored_map = cm(anomaly_map_resized)
        colored_map = (colored_map[:, :, :3] * 255).astype(np.uint8)
        colored_map = cv2.cvtColor(colored_map, cv2.COLOR_RGB2BGR)

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
        alpha = 0.15
        overlayed = cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0)

        text = f"Anomaly | id:{idx}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 2
        text_color = (190, 60, 0)

        # Calcola le dimensioni del testo per centrarlo
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Posizione: Metà larghezza - metà testo, altezza fissa (es. 30px dal bordo superiore)
        text_x = (overlayed.shape[1] - text_width) // 2
        text_y = text_height + 10

        cv2.putText(overlayed, text, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

        return overlayed, mask, colored_map

    # -------------------------------------------------
    # Single image processing
    # -------------------------------------------------
    def _update_statistics(self, cls_token, p_tokens_target):
        """Update internal stats (meams and Cholesky)."""
        self.memory_cls_features.append(cls_token)
        self.memory_patch_features.append(p_tokens_target)
        if len(self.memory_cls_features) >= 2:
            # Update CLS stats
            cls_stack = np.stack(self.memory_cls_features)
            self.cached_mean_cls, self.cached_L_cls = cholesky(cls_stack)

            # Calculate current distance for the threshold buffer
            diff_cls = (cls_token - self.cached_mean_cls).reshape(1, -1)
            dist_cls = np.sqrt(mahalanobis_cholesky(diff_cls, self.cached_L_cls))[0]
            self.good_cls_distances.append(dist_cls)

            # Update Patch stats
            patch_stack = np.concatenate(self.memory_patch_features, axis=0)
            self.cached_mean_patch, self.cached_L_patch = cholesky(patch_stack)

            diff_patch = p_tokens_target - self.cached_mean_patch
            patch_scores = mahalanobis_cholesky(diff_patch, self.cached_L_patch)
            self.good_patch_scores.append(patch_scores)

            #  Dynamic Threshold Update
            p99 = np.percentile(self.good_cls_distances, 99)
            self.threshold = np.round(p99 + np.std(self.good_cls_distances) * 2, 3)

            return True

        return False

    def _process_single_image(self, image: np.ndarray):
        self.global_idx += 1
        idx = self.global_idx

        # Estrazione Features
        x = self._preprocess(image)
        patch_tokens, cls_token = self._extract_features(x)
        num_real_patches = (self.target_img_size // 16) ** 2
        p_tokens_target = patch_tokens[-num_real_patches:]

        dist_cls = 0.0
        patch_scores = np.zeros(num_real_patches, dtype=np.float32)
        anomaly_map_norm = filtered_mask = None
        is_anomaly = False
        total_area = 0.0

        # --- CASE A: INFERENCE MODE (Warmup Done) and update memories if not is anomaly ---
        if self.warmup_done and self.cached_mean_cls is not None:
            # CLS Distance
            diff_cls = (cls_token - self.cached_mean_cls).reshape(1, -1)
            dist_cls = np.sqrt(mahalanobis_cholesky(diff_cls, self.cached_L_cls))[0]

            # Patch Scores
            diff_patch = p_tokens_target - self.cached_mean_patch
            patch_scores = mahalanobis_cholesky(diff_patch, self.cached_L_patch)

            # Anomaly Map Generation
            grid = int(np.sqrt(num_real_patches))
            anomaly_map = patch_scores.reshape(grid, grid)

            # Normalization
            gpsa = np.array(self.good_patch_scores)
            map_min = np.percentile(gpsa, 80, axis=0).reshape(grid, grid)
            map_max = map_min * 5
            anomaly_map_norm = np.clip((anomaly_map - map_min) / (map_max - map_min + 1e-6), 0.0, 1.0)

            # Post-Processing(Resize & Border Cleaning)
            mask_up = cv2.resize(anomaly_map_norm, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_LINEAR)
            b = self.border
            mask_up[:b, :] = mask_up[-b:, :] = mask_up[:, :b] = mask_up[:, -b:] = 0
            mid = mask_up.shape[0] // 2
            mask_up[mid - b : mid + b, :] = 0

            # Connected Components
            binary_mask = (mask_up > self.anomaly_threshold).astype(np.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)

            for s_idx, stat in enumerate(stats[1:], start=1):
                if stat[2] * stat[3] >= self.area_threshold:  # w * h
                    total_area += stat[2] * stat[3]
                    if filtered_mask is None:
                        filtered_mask = np.zeros_like(binary_mask)
                    filtered_mask[labels == s_idx] = 1

            # IS ANOMALY DECISION
            is_anomaly = self.warmup_done and dist_cls > self.threshold and total_area > self.global_area_threshold

            if not is_anomaly:
                self._update_statistics(cls_token, p_tokens_target)

        # --- CASE B: WARMUP MODE (Collecting Statistics) ---
        else:
            self._update_statistics(cls_token, p_tokens_target)

        # --- CASE C: TRANSITION (Finish Warmup) ---
        if not self.warmup_done and len(self.memory_cls_features) >= self.warmup_window_size:
            self.warmup_done = True
            base_path = (
                Path("/opt/aisent/data/dino_states")
                / Path(f"type_{self.listello_tipology}")
                / f"{self.listello_height_mm}_{self.listello_width_mm}"
            )
            base_path.mkdir(parents=True, exist_ok=True)
            file_path = base_path / f"{self.camera_name}.npz"
            thrs = self.threshold

            data_to_save = {
                "path": file_path,
                "memory_cls_features": np.copy(np.stack(self.memory_cls_features)),
                "patch_stack": np.copy(np.concatenate(self.memory_patch_features, axis=0)),
                "good_cls_distances": np.copy(np.array(self.good_cls_distances)),
                "good_patch_scores": np.copy(np.array(self.good_patch_scores)),
                "threshold": thrs,
            }
            # handle_thread(self._save_warmup, kwargs=data_to_save).start()
            self._save_warmup(**data_to_save)

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
    # Loop
    # -------------------------------------------------
    def run(self, image: np.ndarray, camera_id: str):
        out = self._process_single_image(image)

        logger.info(
            f"Cam {camera_id} | Image {out['idx']} | dist={out['dist_cls']:.3f} | "
            f"thresh={out['threshold']:.3f} | anomaly={out['is_anomaly']} | "
            f"warmup_done={self.warmup_done} | memory={len(self.memory_cls_features)} | "
        )

        if out["is_anomaly"]:
            overlayed, mask, colored_map = self._create_anomaly_image_threshold(
                out["idx"],
                out["image"],
                out["anomaly_map_norm"],
                threshold=self.anomaly_threshold,
            )
            return overlayed, mask, colored_map, True

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        colored_map = np.zeros_like(image)

        return image.copy(), mask, colored_map, False
