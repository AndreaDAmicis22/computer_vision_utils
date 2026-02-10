import logging
import random
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import openvino as ov
from matplotlib import pyplot as plt
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s.%(msecs)03d | %(threadName)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("AnomalyDetector")


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
        ramp_start=10,
        ramp_end=40,
        start_gamma=0.3,
        area_threshold=200,
        global_area_threshold=300,
        use_custom_min_max=False,
        anomaly_min=100.0,
        anomaly_max=1300.0,
        target_img_size=512,
        anomaly_threshold=0.6,
    ):
        self.warmup_window_size = warmup_window_size
        self.inference_window_size = inference_window_size
        self.ramp_start = ramp_start
        self.ramp_end = ramp_end
        self.start_gamma = start_gamma
        self.area_threshold = area_threshold
        self.global_area_threshold = global_area_threshold
        self.use_custom_min_max = use_custom_min_max
        self.anomaly_min = anomaly_min
        self.anomaly_max = anomaly_max
        self.anomaly_threshold = anomaly_threshold
        self.target_img_size = target_img_size
        self.gamma = start_gamma

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

        if onnx_model_path is not None:
            self.session = ort.InferenceSession(
                onnx_model_path,
                providers=list(providers),
            )

            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            assert len(self.output_names) == 2, "ONNX model must output (cls, patches)"

        # Sliding window memory
        self.lock = threading.Lock()
        self.memory_cls_features = deque(maxlen=inference_window_size)
        self.memory_patch_features = deque(maxlen=inference_window_size)
        self.good_cls_distances = deque(maxlen=inference_window_size)
        self.good_patch_scores = deque(maxlen=inference_window_size)

        self.cached_mean_cls = None
        self.cached_covinv_cls = None
        self.cached_mean_patch = None
        self.cached_covinv_patch = None

        self.threshold = 0
        self.global_idx = 0
        self.warmup_done = False
        self._warmup_state_exists: bool = False

        assert self.warmup_window_size <= self.inference_window_size, (
            f"Errore di configurazione: warmup_window_size ({self.warmup_window_size}) "
            f"non può essere maggiore di inference_window_size ({self.inference_window_size})"
        )

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
        logger.info(f"Salvato stato in: {path}")

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
        cls_stack = np.stack(list(self.memory_cls_features))
        self.cached_mean_cls, self.cached_covinv_cls = compute_mean_cov_inv(cls_stack)
        patch_stack = self.memory_patch_features[0]
        self.cached_mean_patch, self.cached_covinv_patch = compute_mean_cov_inv(patch_stack)

        self.threshold = float(data["threshold"])
        self.warmup_done = True
        self._warmup_state_exists = True
        logger.info(f"Caricato stato da: {path}")

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

        # In parallel build colored map
        if anomaly_map_resized.max() > 1.0:
            anomaly_map_resized = anomaly_map_resized / 255.0
        cm = plt.get_cmap("inferno")
        colored_map = cm(anomaly_map_resized)

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
        alpha = 0.3
        overlayed = cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0)
        return overlayed, mask, colored_map

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    def fit_all(self, images_path: str, save_path: str):
        """Esegue il training su tutte le immagini presenti nella cartella."""
        image_files = self._get_image_files(images_path)
        return self._fit(image_files, save_path)

    def fit_sample(self, images_path: str, save_path: str, n: int = 100):
        """Esegue il training su un campione casuale di N immagini."""
        image_files = self._get_image_files(images_path)

        if len(image_files) > n:
            logger.info(f"Sampling attivo: selezione casuale di {n} immagini su {len(image_files)}.")
            image_files = random.sample(image_files, n)
        else:
            logger.info(
                f"Campionamento richiesto ({n}), ma trovate solo {len(image_files)} immagini. Procedo con tutte."
            )

        return self._fit(image_files, save_path)

    def _get_image_files(self, images_path: str) -> list:
        """Helper interno per recuperare i path delle immagini."""
        path = Path(images_path)
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        files = [f for f in path.iterdir() if f.suffix.lower() in valid_extensions]

        if not files:
            msg = f"Nessuna immagine trovata in {images_path}"
            raise FileNotFoundError(msg)
        return files

    def _fit(self, image_files: list, save_path: str):
        """
        Metodo core che esegue il training effettivo su una lista di file.
        """
        save_path = Path(save_path)

        for img_path in tqdm(image_files, desc="Training"):
            try:
                image_bgr = cv2.imread(str(img_path))
                if image_bgr is None:
                    logger.info(f"Impossibile leggere: {img_path}")
                    continue

                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

                x = self._preprocess(image_rgb)
                patch_tokens, cls_token = self._extract_features(x)

                if len(self.memory_cls_features) >= 2:
                    cls_stack = np.stack(self.memory_cls_features)
                    mean_cls, covinv_cls = compute_mean_cov_inv(cls_stack)
                    dist_cls = mahalanobis_distance(cls_token[None, :], mean_cls, covinv_cls)[0]

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
                logger.info(f"Errore durante l'elaborazione di {img_path}: {e}")
                continue

        # --- Calcolo soglie e salvataggio ---
        if self.good_cls_distances:
            std_cls = np.std(self.good_cls_distances)
            p99 = np.percentile(self.good_cls_distances, 99)
            self.threshold = p99 + std_cls

            self._save_warmup(path=save_path)
            self.warmup_done = True
            logger.info(f"Warmup completato con {len(self.memory_cls_features)} immagini.")
        else:
            logger.info("Errore: Nessuna feature estratta correttamente. Training fallito.")

    # -------------------------------------------------
    # Single image processing
    # -------------------------------------------------
    def _process_single_image(self, image: np.ndarray):
        with self.lock:
            self.global_idx += 1
            idx = self.global_idx
            warmup_done = self.warmup_done
            m_cls, c_cls = self.cached_mean_cls, self.cached_covinv_cls
            m_patch, c_patch = self.cached_mean_patch, self.cached_covinv_patch
            a_min, a_max = self.anomaly_min, self.anomaly_max
            curr_threshold = self.threshold

        x = self._preprocess(image)
        patch_tokens, cls_token = self._extract_features(x)
        num_real_patches = (self.target_img_size // 16) ** 2
        p_tokens_target = patch_tokens[-num_real_patches:]

        dist_cls = 0.0
        total_area = 0.0
        anomaly_map_norm = None
        filtered_mask = None
        patch_scores = np.zeros(num_real_patches, dtype=np.float32)

        if m_cls is not None and m_patch is not None and warmup_done:
            # Inferenza classica
            # logger.info(f"Memory used! {idx}")
            # CLS
            dist_cls = mahalanobis_distance(cls_token[None, :], m_cls, c_cls)[0]

            # Patch
            diff = p_tokens_target - m_patch
            patch_scores = np.einsum("nd,df,nf->n", diff, c_patch, diff)
            grid = int(np.sqrt(num_real_patches))
            anomaly_map = patch_scores.reshape(grid, grid)

            # Normalizzazione (usando min/max correnti)
            anomaly_map_norm = np.clip((anomaly_map - a_min) / (a_max - a_min + 1e-6), 0.0, 1.0)

            # Resize e Componenti connesse
            anomaly_map_up = cv2.resize(anomaly_map_norm, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_LINEAR)
            binary_mask_np = (anomaly_map_up > self.anomaly_threshold).astype(np.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_np, 8)

            for stat_idx, stat in enumerate(stats[1:], start=1):
                if stat[2] * stat[3] >= self.area_threshold:  # w * h
                    total_area += stat[4]  # area effettiva
                    if filtered_mask is None:
                        filtered_mask = np.zeros_like(binary_mask_np)
                    filtered_mask[labels == stat_idx] = 1

            is_anomaly = self.warmup_done and dist_cls > curr_threshold and total_area > self.global_area_threshold

            # Lanciamo l'aggiornamento in background e usciamo subito col return
            threading.Thread(
                target=self._post_process_update,
                args=(cls_token, p_tokens_target, dist_cls, is_anomaly, patch_scores),
                daemon=True,
            ).start()

        else:
            # WARM UP
            with self.lock:
                self.memory_cls_features.append(cls_token)
                self.memory_patch_features.append(p_tokens_target)

                if len(self.memory_cls_features) >= 2:
                    cls_stack = np.stack(self.memory_cls_features)
                    self.cached_mean_cls, self.cached_covinv_cls = compute_mean_cov_inv(cls_stack)

                    patch_stack = np.concatenate(self.memory_patch_features, axis=0)
                    self.cached_mean_patch, self.cached_covinv_patch = compute_mean_cov_inv(patch_stack)

                    dist_cls = mahalanobis_distance(cls_token[None, :], self.cached_mean_cls, self.cached_covinv_cls)[0]
                    diff = p_tokens_target - self.cached_mean_patch
                    patch_scores = np.einsum("nd,df,nf->n", diff, self.cached_covinv_patch, diff)

                    self.good_cls_distances.append(dist_cls)
                    self.good_patch_scores.extend(patch_scores)

                # Check fine warmup interno al lock
                if not self.warmup_done and len(self.memory_cls_features) >= self.warmup_window_size:
                    self.warmup_done = True
                    self._save_warmup(path="/workspace/src/SPA006/new_states/state_running.npz")
                # logger.info(f"LEN: {len(self.memory_cls_features)}")
            is_anomaly = False

        # --- FASE 2: PREPARAZIONE RISPOSTA ---
        return {
            "idx": idx,
            "is_anomaly": is_anomaly,
            "dist_cls": dist_cls,
            "image": image,
            "threshold": self.threshold,
            "anomaly_map_norm": anomaly_map_norm,
            "binary_mask": filtered_mask,
        }

    def _post_process_update(self, cls_token, patch_tokens, dist_cls, is_anomaly, patch_scores):
        with self.lock:
            # 1. Aggiornamento Memoria (solo se non è anomalo e siamo in warmup)
            if not is_anomaly:
                self.memory_cls_features.append(cls_token)
                self.memory_patch_features.append(patch_tokens)
                self.good_cls_distances.append(dist_cls)
                self.good_patch_scores.extend(patch_scores)

            # 2. Ricalcolo Statistiche (Cache)
            # Calcoliamo mean/covinv solo se abbiamo abbastanza campioni
            if len(self.memory_cls_features) >= 2:
                # CLS
                cls_stack = np.stack(self.memory_cls_features)
                self.cached_mean_cls, self.cached_covinv_cls = compute_mean_cov_inv(cls_stack)

                # Patch
                patch_stack = np.concatenate(self.memory_patch_features, axis=0)
                self.cached_mean_patch, self.cached_covinv_patch = compute_mean_cov_inv(patch_stack)

            # 3. Aggiornamento Dinamico Soglia (Threshold) + Ramping
            if len(self.good_cls_distances) > 1:
                p99 = np.percentile(self.good_cls_distances, 99)
                std = np.std(self.good_cls_distances)
                if self.global_idx <= self.ramp_start:
                    self.gamma = self.start_gamma
                else:
                    valore_calcolato = self.start_gamma + (1.0 - self.start_gamma) * (
                        (self.global_idx - self.ramp_start) / (self.ramp_end - self.ramp_start)
                    )
                    self.gamma = np.round(np.clip(valore_calcolato, self.start_gamma, 1.0), 3)

                self.threshold = np.round(self.gamma * p99 + std, 3)

            if not self.use_custom_min_max and len(self.good_patch_scores) > self.inference_window_size:
                p99_patch = np.percentile(self.good_patch_scores, 99)
                self.anomaly_max = p99_patch * 2 + np.std(self.good_patch_scores)
                self.anomaly_min = p99_patch
            # 5. Check fine Warmup
            if not self.warmup_done and len(self.memory_cls_features) >= self.warmup_window_size:
                self.warmup_done = True
                self._save_warmup(path="/home/aisent/Desktop/dev/SPA006/new_states/state_running.npz")
            # logger.info(f"Memory updated! {self.global_idx}")
            # logger.info(f"LEN: {len(self.memory_cls_features)}")

    # -------------------------------------------------
    # Sliding window loop
    # -------------------------------------------------
    def run(self, image: np.ndarray):
        out = self._process_single_image(image)

        logger.info(
            f"Image {out['idx']} | dist={out['dist_cls']:.3f} | "
            f"thresh={self.threshold} | anomaly={out['is_anomaly']} | "
            f"warmup_done={self.warmup_done} | memory={len(self.memory_cls_features)} | "
            f"gamma={self.gamma} | "
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
