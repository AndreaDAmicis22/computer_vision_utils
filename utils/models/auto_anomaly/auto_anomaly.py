import logging
from pathlib import Path

import cv2
import numpy as np
from utils.models.auto_anomaly.dino_onnx import SlidingWindowAnomalyDetectorONNX

# from packages.redis_utils import redis_utils

logger = logging.getLogger(__name__)


class AutoAnomaly:
    def __init__(
        self,
        warmup_window_size: int = 12,
        inference_window_size: int = 12,
        area_threshold: int = 1200,
        global_area_threshold: int = 300,
        target_img_size: int = 512,
        ramp_start: int = 32,
        ramp_end: int = 100,
        anomaly_threshold: float = 0.6,
    ):
        self.swad = SlidingWindowAnomalyDetectorONNX(
            onnx_model_path="/workspace/src/utils/utils/models/auto_anomaly/assets/vit_large_dinov3_features.onnx",
            warmup_window_size=warmup_window_size,
            inference_window_size=inference_window_size,
            area_threshold=area_threshold,
            global_area_threshold=global_area_threshold,
            target_img_size=target_img_size,
            ramp_start=ramp_start,
            ramp_end=ramp_end,
            anomaly_threshold=anomaly_threshold,
        )
        self.warmup_window_size = warmup_window_size
        self.area_threshold = area_threshold
        self.warmup_counter = 0

    def _state_exists(self, path: Path):
        if not path.exists():
            path.mkdir()
            logger.info("State does not exist. Creating folder")
            return False, ""
        files = list(path.glob("*.npz"))
        if len(files) > 0:
            return True, files[0]
        return False, ""

    def try_load_warmup(self, path: str):
        path = Path(path)
        exists, state_path = self._state_exists(path)
        if exists:
            self.swad.load_warmup(state_path)
            logger.info("State loaded")
            self.warmup_counter = self.warmup_window_size

    def _preprocess(self, image: np.ndarray, gamma=1.0):
        return self.gamma_correction(image, gamma)

    def fit(self, images_path: str, save_path: str):
        self.swad.fit(images_path, save_path)

    def predict(self, image: np.ndarray):
        self.try_load_warmup("/workspace/src/SPA006/")
        out_img, drawn_contours, is_anomalous = self.swad.run(image)
        if self.warmup_counter < self.warmup_window_size:
            self.warmup_counter += 1
            return image, False
        return out_img, drawn_contours, is_anomalous

    def gamma_correction(self, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(self, table)
