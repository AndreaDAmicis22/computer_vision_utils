from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .feature_extractor import FeatureExtractor
from .metrics_computer import MetricsComputer


@dataclass
class Config:
    warmup_window_size: int
    target_image_size: int
    feature_extractor: FeatureExtractor
    metrics_computer: MetricsComputer


@dataclass
class DynamicParameters:
    area_threshold: int
    anomaly_threshold: float
    disc_threshold: float


@dataclass(frozen=True)
class SStadPrediction:
    heatmap: np.ndarray
    defects_mask: np.ndarray
    is_anomalous: bool


class StaticStatefulAnomalyDetector:
    def __init__(
        self,
        configuration: Config,
        dynamic_parameters: DynamicParameters,
    ):
        self._warmup_window_size = configuration.warmup_window_size
        self._target_image_size = configuration.target_image_size
        self._FE = configuration.feature_extractor
        self._MC = configuration.metrics_computer

        self._ANOMALY_THRESHOLD = dynamic_parameters.anomaly_threshold
        self._AREA_THRESHOLD = dynamic_parameters.area_threshold
        self._DISC_THRESHOLD = dynamic_parameters.disc_threshold

        self._memory_cls_features = deque(maxlen=configuration.warmup_window_size)
        self._memory_patch_features = deque(maxlen=configuration.warmup_window_size)
        self._good_cls_distances = deque(maxlen=configuration.warmup_window_size)
        self._good_patch_scores = deque(maxlen=configuration.warmup_window_size)

    def save_state(self, path: Path):
        np.savez_compressed(
            file=path,
            cls_stack=self._cls_stack,
            patch_stack=self._patch_stack,
            anomaly_threshold=self._ANOMALY_THRESHOLD,
            area_threshold=self._AREA_THRESHOLD,
            disc_threshold=self._DISC_THRESHOLD,
        )

    def load_state(self, path):
        data = np.load(path)

        self._cls_stack = data["cls_stack"]
        self._patch_stack = data["patch_stack"]

        self._DISC_THRESHOLD = float(data["disc_threshold"])
        self._ANOMALY_THRESHOLD = float(data["anomaly_threshold"])
        self._AREA_THRESHOLD = int(data["area_threshold"])

    def update_dynamic_parameters(self, new_params: DynamicParameters):
        self._ANOMALY_THRESHOLD = new_params.anomaly_threshold
        self._AREA_THRESHOLD = new_params.area_threshold
        self._DISC_THRESHOLD = new_params.disc_threshold

    def _process_image(self, image):
        patch_tokens, cls_token = self._FE.extract_features(image, target_image_size=self._target_image_size)
        cls_score = None
        patch_scores = None
        if len(self._memory_cls_features) >= 2:
            cls_score = self._MC.compute_cls_score(self._cls_stack, cls_token)

            patch_scores = self._MC.compute_patch_scores(self._patch_stack, self._target_image_size, patch_tokens)
        return cls_token, cls_score, patch_tokens, patch_scores

    def warmup(self, image: np.ndarray):
        cls_token, cls_score, patch_tokens, patch_scores = self._process_image(image)

        self._memory_cls_features.append(cls_token)
        self._memory_patch_features.append(patch_tokens)
        if len(self._memory_cls_features) >= 2:
            self._good_cls_distances.append(cls_score)
            self._good_patch_scores.append(patch_scores)

            self._cls_stack = np.stack(self._memory_cls_features)
            self._patch_stack = np.concatenate(self._memory_patch_features, axis=0)

    def _create_heatmap(self, patch_scores: np.ndarray) -> np.ndarray:
        grid = int(np.sqrt((self._target_image_size // 16) ** 2))
        reshaped_patch_scores = patch_scores.reshape(grid, grid)

        gpsa = np.array(self._good_patch_scores)

        mat_p5 = np.max(gpsa, axis=0).reshape(grid, grid)
        mat_p95 = mat_p5 * 5

        downscaled_heatmap = np.clip((reshaped_patch_scores - mat_p5) / (mat_p95 - mat_p5 + 1e-6), 0.0, 1.0)
        return cv2.resize(
            downscaled_heatmap,
            (
                self._target_image_size,
                self._target_image_size,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

    def _create_defects_mask(self, heatmap: np.ndarray) -> np.ndarray:
        binary_mask = (heatmap > self._ANOMALY_THRESHOLD).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)

        total_area = 0
        filtered_mask = np.zeros_like(binary_mask)
        for stat_idx, stat in enumerate(stats[1:], start=1):
            _, _, w, h, _area = stat
            if w * h >= self._AREA_THRESHOLD:
                filtered_mask[labels == stat_idx] = 1
                total_area += w * h

        return filtered_mask

    def _is_defected(self, cls_score: float) -> bool:
        return cls_score > self._DISC_THRESHOLD

    def predict(self, image: np.ndarray) -> SStadPrediction:
        _, cls_score, _, patch_scores = self._process_image(image)
        heatmap = self._create_heatmap(patch_scores)
        defects_mask = self._create_defects_mask(heatmap)
        return SStadPrediction(heatmap, defects_mask, self._is_defected(cls_score))
