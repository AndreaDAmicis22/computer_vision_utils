import logging
from pathlib import Path

import cv2
import numpy as np
from utils.models.auto_anomaly.dino_onnx_linear import SlidingWindowAnomalyDetectorONNX

# from packages.redis_utils import redis_utils

logger = logging.getLogger(__name__)


class AutoAnomaly:
    def __init__(
        self,
        onnx_model_path: str = "/home/aisent/Desktop/dev/utils/utils/models/auto_anomaly/assets/vit_base_patch16_dinov3.lvd1689m_400.onnx",
        warmup_window_size: int = 15,
        inference_window_size: int = 30,
        area_threshold: int = 1200,
        global_area_threshold: int = 300,
        target_img_size: int = 512,
        start_gamma: float = 0.3,
        ramp_start: int = 10,
        ramp_end: int = 40,
        anomaly_threshold: float = 0.6,
        use_custom_min_max: bool = False,
        anomaly_min: int = 100,
        anomaly_max: int = 1300,
        name_state: str = "state",
    ):
        """
        Initializes the auto-anomaly detector based on Sliding Window and ONNX.

        Args:
            warmup_window_size (int): Number of initial frames/images used to establish the baseline.
            inference_window_size (int): Size of the temporal window for continuous inference.
            area_threshold (int): Minimum pixel area for a local anomaly to be considered valid.
            global_area_threshold (int): Minimum total pixel area (sum) required across the entire image.
            target_img_size (int): Side length to which images are resized before processing.
            ramp_start (int): Starting value for the anomaly ramping (convergence param).
            ramp_end (int): End value for the anomaly ramping (convergence param).
            anomaly_threshold (float): Confidence threshold (0-1) above which is thresholded the filterd anomaly map.
            use_custom_min_max (bool): flag to use custom min/max normalization.
            anomaly_min (int): min parameter of anomaly map normalization.
            anomaly_max (int): max parameter of anomaly map normalization.
        """
        self.swad = SlidingWindowAnomalyDetectorONNX(
            onnx_model_path=onnx_model_path,
            warmup_window_size=warmup_window_size,
            inference_window_size=inference_window_size,
            area_threshold=area_threshold,
            global_area_threshold=global_area_threshold,
            target_img_size=target_img_size,
            start_gamma=start_gamma,
            ramp_start=ramp_start,
            ramp_end=ramp_end,
            anomaly_threshold=anomaly_threshold,
            use_custom_min_max=use_custom_min_max,
            anomaly_min=anomaly_min,
            anomaly_max=anomaly_max,
            name_state=name_state,
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

    def _file_exists(self, file_path: Path) -> bool:
        """Verifica se il file specifico esiste ed è un file .npz"""
        return bool(file_path.exists() and file_path.is_file() and file_path.suffix == ".npz")

    def load_warmup_from_path(self, full_path: str):
        """Carica lo stato da un percorso file completo"""
        path = Path(full_path)

        if self._file_exists(path):
            self.swad.load_warmup(path)
            logger.info(f"State loaded from {path}")
            self.warmup_counter = self.warmup_window_size
        else:
            logger.error(f"File not found or invalid: {full_path}")

    def _preprocess(self, image: np.ndarray, gamma=1.0):
        return self.gamma_correction(image, gamma)

    def fit_sample(self, images_path: str, save_path: str, n: int = 100):
        """
        Performs warmup or training by sampling the images and saving the model state.

        Args:
            images_path (str | Path): Path to the directory containing the images.
            save_path (str | Path): Path to the file where the state will be saved (e.g., '/path/to/state.npz').
            n (int): Number of images to sample when `make_sample` is set to True.
        """
        self.swad.fit_sample(images_path, save_path, n)

    def fit_all(self, images_path: str, save_path: str):
        """
        Performs warmup or training by extracting features from images and saving the model state.

        Args:
            images_path (str | Path): Path to the directory containing the images.
            save_path (str | Path): Path to the file where the state will be saved (e.g., '/path/to/state.npz').
        """
        self.swad.fit_all(images_path, save_path)

    def predict(self, image: np.ndarray):
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
        out_img, drawn_contours, colored_map, is_anomalous = self.swad.run(image)
        if self.warmup_counter < self.warmup_window_size:
            self.warmup_counter += 1
            return out_img, drawn_contours, colored_map, False
        return out_img, drawn_contours, colored_map, is_anomalous

    def gamma_correction(self, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        return cv2.LUT(self, table)
