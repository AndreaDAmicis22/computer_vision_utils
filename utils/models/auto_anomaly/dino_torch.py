import logging
import random
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as T
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s.%(msecs)03d | %(threadName)s | %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("AnomalyDetector")

MODEL_SIZE_MAP = {
    "h": "vit_huge_plus_patch16_dinov3.lvd1689m",
    "l": "vit_large_patch16_dinov3.lvd1689m",
    "b": "vit_base_patch16_dinov3.lvd1689m",
    "sp": "vit_small_plus_patch16_dinov3.lvd1689m",
    "s": "vit_small_patch16_dinov3.lvd1689m",
}


# -------------------------------------------------
# Math utilities (Torch)
# -------------------------------------------------
def compute_mean_cov_inv(features: torch.Tensor | np.ndarray, eps: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor]:
    features = torch.tensor(features, dtype=torch.float32)
    mean = torch.mean(features, dim=0)
    diffs = features - mean
    cov = torch.matmul(diffs.T, diffs) / (features.shape[0] - 1)
    cov += eps * torch.eye(cov.shape[0])
    cov_inv = torch.linalg.inv(cov)
    return mean, cov_inv


def mahalanobis_distance(vec: torch.Tensor, mean: torch.Tensor, cov_inv: torch.Tensor) -> torch.Tensor:
    diff = vec - mean
    left = torch.matmul(diff, cov_inv)
    d_sq = torch.sum(left * diff, dim=1)
    return torch.sqrt(d_sq)


# -------------------------------------------------
# Sliding Window Anomaly Detector (ONNX)
# -------------------------------------------------
class SlidingWindowAnomalyDetector:
    def __init__(
        self,
        model_size="b",
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
        name_state="state",
    ):
        self.model_size = model_size
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
        self.name_state = name_state

        # Sliding window memory
        self.memory_cls_features = []
        self.memory_patch_features = []
        self.good_cls_distances = []
        self.good_patch_scores = []

        self.cached_mean_cls = None
        self.cached_covinv_cls = None
        self.cached_mean_patch = None
        self.cached_covinv_patch = None

        self.threshold = 0
        self.global_idx = 0
        self.warmup_done = False
        self._warmup_state_exists: bool = False

        self.border = 80
        self.extractor = self._build_extractor()

        assert self.warmup_window_size <= self.inference_window_size, (
            f"Errore di configurazione: warmup_window_size ({self.warmup_window_size}) "
            f"non può essere maggiore di inference_window_size ({self.inference_window_size})"
        )

    # --------------------------
    # Feature extractor
    # --------------------------
    def _build_extractor(self) -> torch.nn.Module:
        model = timm.create_model(MODEL_SIZE_MAP[self.model_size], pretrained=True)
        model.eval()
        return model

    def _extract_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            features = self.extractor.forward_features(x)
            patch_tokens = features[:, 1:, :]  # without CLS
            cls_token = features[:, 0, :]
        return patch_tokens, cls_token

    # -------------------------------------------------
    # Preprocessing
    # -------------------------------------------------
    def _preprocess(self, img_pil: Image.Image) -> torch.Tensor:
        transform = T.Compose(
            [
                T.Resize(size=(self.target_img_size, self.target_img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        return transform(img_pil).unsqueeze(0)

    # -------------------------------------------------
    # Feature extraction via ONNX
    # -------------------------------------------------
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
        self.memory_cls_features = data["memory_cls_features"]
        self.memory_patch_features = [data["patch_stack"]]
        self.good_cls_distances = data["good_cls_distances"]
        self.good_patch_scores = data["good_patch_scores"]
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

        # Bordi
        H, _W = anomaly_map_resized.shape
        anomaly_map_resized[: self.border, :] = 0  # top
        anomaly_map_resized[-self.border :, :] = 0  # bottom
        anomaly_map_resized[:, : self.border] = 0  # left
        anomaly_map_resized[:, -self.border :] = 0  # right
        # Banda orizzontale centrale
        mid = H // 2
        anomaly_map_resized[mid - self.border : mid + self.border, :] = 0

        # In parallel build colored map
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
        alpha = 0.25
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
                image_pil = Image.fromarray(image_rgb)

                x = self._preprocess(image_pil)
                patch_tokens, cls_token = self._extract_features(x)

                if len(self.memory_cls_features) >= 2:
                    cls_stack = torch.stack(self.memory_cls_features)
                    mean_cls, covinv_cls = compute_mean_cov_inv(cls_stack)
                    dist_cls = mahalanobis_distance(cls_token, mean_cls, covinv_cls).item()

                    patch_stack = torch.cat(self.memory_patch_features, dim=0)

                    mean_patch, covinv_patch = compute_mean_cov_inv(patch_stack)
                    num_real_patches = (self.target_img_size // 16) ** 2
                    patch_tokens = patch_tokens[-num_real_patches:]
                    diff = patch_tokens - mean_patch
                    patch_scores = torch.einsum("nd,df,nf->n", diff, covinv_patch, diff)
                    self.good_patch_scores.append(patch_scores)
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
            # std_cls = np.std(self.good_cls_distances)
            p99 = np.percentile(self.good_cls_distances, 95)
            # self.threshold = p99 + std_cls
            self.threshold = p99

            self._save_warmup(path=save_path)
            self.warmup_done = True
            logger.info(f"Warmup completato con {len(self.memory_cls_features)} immagini.")
        else:
            logger.info("Errore: Nessuna feature estratta correttamente. Training fallito.")

    # -------------------------------------------------
    # Single image processing
    # -------------------------------------------------
    def _process_single_image(self, image: np.ndarray):
        self.global_idx += 1
        idx = self.global_idx

        # convert directly to RGB tensor
        if isinstance(image, Image.Image):
            x = self._preprocess(image)
        elif isinstance(image, np.ndarray):
            if image.ndim == 2:  # grayscale
                image = np.stack([image] * 3, axis=-1)
            elif image.shape[-1] == 1:
                image = np.concatenate([image] * 3, axis=-1)
            elif image.shape[-1] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            x = self._preprocess(Image.fromarray(image.astype(np.uint8)))
        elif isinstance(image, torch.Tensor):
            x = image if image.ndim == 4 else image.unsqueeze(0)
        else:
            msg = f"Unsupported image type: {type(image)}"
            raise TypeError(msg)

        # Feature extraction
        patch_tokens, cls_token = self._extract_features(x)
        patch_tokens = patch_tokens.squeeze(0)  # (N, D)

        num_real_patches = (self.target_img_size // 16) ** 2
        p_tokens_target = patch_tokens[-num_real_patches:]

        # Inizializzazione variabili output
        dist_cls = 0.0
        patch_scores = np.zeros(num_real_patches, dtype=np.float32)
        anomaly_map_norm = filtered_mask = None
        is_anomaly = False
        total_area = 0.0

        # 2. Logica di Inferenza (se il modello è pronto)
        if self.warmup_done and self.cached_mean_cls is not None:
            # Calcolo Distanze (CLS & Patch)
            dist_cls = mahalanobis_distance(cls_token, self.cached_mean_cls, self.cached_covinv_cls).item()
            diff = p_tokens_target - self.cached_mean_patch.unsqueeze(0)
            patch_scores = torch.einsum("nd,df,nf->n", diff, self.cached_covinv_patch, diff)

            # Generazione Mappa Anomalia
            grid = int(np.sqrt(num_real_patches))
            anomaly_map = patch_scores.view(grid, grid)
            anomaly_map_norm = (
                torch.clamp((anomaly_map - self.anomaly_min) / (self.anomaly_max - self.anomaly_min + 1e-6), 0.0, 1.0)
                .detach()
                .cpu()
                .numpy()
            )

            # Post-Processing Maschera (Resize & Border Cleaning)
            mask_up = cv2.resize(anomaly_map_norm, (x.shape[3], x.shape[2]), interpolation=cv2.INTER_LINEAR)
            b = self.border
            mask_up[:b, :] = mask_up[-b:, :] = mask_up[:, :b] = mask_up[:, -b:] = 0
            mid = mask_up.shape[0] // 2
            mask_up[mid - b : mid + b, :] = 0

            # Componenti Connesseanomaly_map_up
            binary_mask = (mask_up > self.anomaly_threshold).astype(np.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)

            for s_idx, stat in enumerate(stats[1:], start=1):
                if stat[2] * stat[3] >= self.area_threshold:  # w * h
                    total_area += stat[2] * stat[3]
                    if filtered_mask is None:
                        filtered_mask = np.zeros_like(binary_mask)
                    filtered_mask[labels == s_idx] = 1

            # Decisione Finale
            is_anomaly = self.warmup_done and dist_cls > self.threshold and total_area > self.global_area_threshold

        # 3. Aggiornamento Memoria e Statistiche (Background logic logicamente sequenziale)
        if not is_anomaly:
            if len(self.memory_cls_features) >= self.inference_window_size:
                self.memory_cls_features.pop(0)
                self.memory_patch_features.pop(0)
                self.good_cls_distances.pop(0)
                self.good_patch_scores.pop(0)

            self.memory_cls_features.append(cls_token.squeeze(0))
            self.memory_patch_features.append(p_tokens_target.squeeze(0))
            # Se siamo in warmup, le distanze vengono calcolate solo se abbiamo almeno 2 campioni
            if len(self.memory_cls_features) >= 2:
                # Update Cache Statistiche
                cls_stack = torch.stack(self.memory_cls_features)
                self.cached_mean_cls, self.cached_covinv_cls = compute_mean_cov_inv(cls_stack)
                patch_stack = torch.cat(self.memory_patch_features, dim=0)
                self.cached_mean_patch, self.cached_covinv_patch = compute_mean_cov_inv(patch_stack)

                # Se non eravamo in inferenza, calcoliamo distanze per lo storico soglia
                if not self.warmup_done:
                    dist_cls = mahalanobis_distance(cls_token, self.cached_mean_cls, self.cached_covinv_cls).item()
                    patch_scores = torch.einsum(
                        "nd,df,nf->n",
                        p_tokens_target - self.cached_mean_patch,
                        self.cached_covinv_patch,
                        p_tokens_target - self.cached_mean_patch,
                    )

                self.good_cls_distances.append(dist_cls)
                self.good_patch_scores.append(patch_scores)

            # Update Dinamico Soglia & Ramping
            if len(self.good_cls_distances) > 1:
                p99 = np.percentile(self.good_cls_distances, 75)
                progress = np.clip((self.global_idx - self.ramp_start) / (self.ramp_end - self.ramp_start + 1e-6), 0, 1)
                self.gamma = (
                    np.round(self.start_gamma + (1.0 - self.start_gamma) * progress, 3)
                    if self.global_idx > self.ramp_start
                    else self.start_gamma
                )
                self.threshold = np.round(self.gamma * p99, 3)

            # Update Min/Max per normalizzazione mappa
            if not self.use_custom_min_max and len(self.good_patch_scores) > self.inference_window_size:
                self.anomaly_min = np.max(self.good_patch_scores)
                self.anomaly_max = np.percentile(self.good_patch_scores, 99) * 2 + np.std(self.good_patch_scores)

        # 4. Gestione Fine Warmup
        if not self.warmup_done and len(self.memory_cls_features) >= self.warmup_window_size:
            self.warmup_done = True
            self._save_warmup(path=f"/workspace/src/SPA006/new_states/{self.name_state}_running.npz")

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
