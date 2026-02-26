import logging
from pathlib import Path

import cv2
import numpy as np
from cv_pack.models.auto_anomaly_static_cholesky.dino_onnx import AnomalyDetectorONNX

# from packages.redis_utils import redis_utils

logger = logging.getLogger(__name__)


class AutoAnomaly:
    def __init__(
        self,
        onnx_model_path: str = "/home/aisent/Desktop/dev/utils/utils/models/auto_anomaly/assets/vit_base_patch16_dinov3.lvd1689m_400.onnx",
        warmup_window_size: int = 15,
        target_img_size: int = 512,
        anomaly_threshold: float = 0.6,
        area_threshold: int = 10,
        global_area_threshold: int = 100,
        camera_name: str = "camera",
    ):
        """
        Initializes the auto-anomaly detector based on Sliding Window and ONNX.

        Args:
            warmup_window_size (int): Number of initial frames/images used to establish the baseline.
            inference_window_size (int): Size of the temporal window for continuous inference.
            area_threshold (int): Minimum pixel area for a local anomaly to be considered valid.
            global_area_threshold (int): Minimum total pixel area (sum) required across the entire image.
            target_img_size (int): Side length to which images are resized before processing.
            anomaly_threshold (float): Confidence threshold (0-1) above which is thresholded the filterd anomaly map.
        """
        self.swad = AnomalyDetectorONNX(
            onnx_model_path=onnx_model_path,
            warmup_window_size=warmup_window_size,
            area_threshold=area_threshold,
            global_area_threshold=global_area_threshold,
            target_img_size=target_img_size,
            anomaly_threshold=anomaly_threshold,
            camera_name=camera_name,
        )
        self.camera_name = camera_name
        self.warmup_window_size = warmup_window_size
        self.area_threshold = area_threshold
        self.warmup_counter = 0

    def _state_exists(self, path: Path):
        if not path.exists():
            path.mkdir()
            logger.info("State does not exist. Creating folder")
            return False, ""
        files = list(path.glob(f"{self.camera_name}.npz"))
        if len(files) > 0:
            return True, files[0]
        return False, ""

    def _state_exists(self, path: Path):
        try:
            if not path.exists():
                path.mkdir(exist_ok=True)
                logger.info(f"📂 State directory created: {path}")
                return False, ""

        except OSError as e:
            logger.exception(f"❌ Cannot create directory {path}: {e}")
            return False, ""
        try:
            files = list(path.glob(f"{self.camera_name}.npz"))
            if len(files) > 0:
                return True, files[0]
        except Exception as e:
            logger.warning(f"⚠️ Error searching for checkpoint files: {e}")
        return False, ""

    def try_load_warmup(self):
        path = Path(f"/workspace/src/SPA006/dino_states/{self.camera_name}")
        exists, state_path = self._state_exists(path)
        if exists:
            self.swad.load_warmup(state_path)
            logger.info("State loaded")
            self.warmup_counter = self.warmup_window_size

    def _normalize(self, image: np.ndarray):
        img = image.astype(np.float32)
        min_val = img.min()
        max_val = img.max()
        img_norm = (img - min_val) / (max_val - min_val)
        return (img_norm * 255).astype(np.uint8)

    def predict(self, image: np.ndarray, camera_id: str):
        """
        Performs anomaly detection on a single input image.

        This method handles the internal warmup counter. Until the warmup period
        (defined by `warmup_window_size`) is completed, the method will always
        return False for the anomaly status to ensure the baseline is stable.

        Args:
            image (np.ndarray): The input image to be analyzed (typically BGR/RGB).

        Returns:
            out_img (np.ndarray): The processed image or visualization.
            drawn_contours (list): List of detected contours around anomalous regions.
            colored_map (np.ndarray): The anomaly heatmap (e.g., using the Inferno colormap).
            is_anomalous (bool): Boolean flag indicating if an anomaly was detected.
                                 Always False during the warmup phase.
        """
        out_img, drawn_contours, colored_map, is_anomalous = self.swad.run(image, camera_id)
        if self.warmup_counter < self.warmup_window_size:
            self.warmup_counter += 1
            return out_img, drawn_contours, colored_map, False
        return out_img, drawn_contours, colored_map, is_anomalous

    def gamma_correction(self, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(self, table)
