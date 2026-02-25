import logging
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv
import numpy as np
from stateful_auto_anomaly.static_stad import (
    Config,
    DynamicParameters,
    StaticStatefulAnomalyDetector,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    output_image: np.ndarray
    is_anomalous: bool


class AutoAnomalyExample:
    def __init__(
        self,
        configuration: Config,
        dynamic_parameters: DynamicParameters,
    ):
        self.stad = StaticStatefulAnomalyDetector(
            configuration=configuration, dynamic_parameters=dynamic_parameters
        )

        self._area_threshold = dynamic_parameters.area_threshold
        self._warmup_window_size = configuration.warmup_window_size
        self._warmup_counter = 0

        self._warmup_saved = False

    def _state_exists(self, path: Path) -> tuple[bool, Path]:
        if not path.exists():
            path.mkdir()
            logger.info("State does not exist. Creating folder")
            return False, Path("")
        files = list(path.glob("state.npz"))
        if len(files) > 0:
            return True, files[0]
        else:
            return False, Path("")

    def try_load_state(self, path: Path):
        exists, state_path = self._state_exists(path)
        if exists:
            self.stad.load_state(state_path)
            logger.info("State loaded")
            self._warmup_counter = self._warmup_window_size
            self._warmup_saved = True

    def _draw_anomaly(self, drawn_contours, image_to_output) -> np.ndarray:
        overlay = np.zeros_like(image_to_output, dtype=np.uint8)
        mask_bool = drawn_contours > 0
        overlay[mask_bool] = (0, 255, 0)
        alpha = 0.2
        image_to_output = cv.addWeighted(overlay, alpha, image_to_output, 1 - alpha, 0)
        return image_to_output

    def _warmup_done(self) -> bool:
        return self._warmup_counter >= self._warmup_window_size

    def _check_warmup_saved(self, path: Path):
        if not self._warmup_saved:
            self.stad.save_state(path)
            self._warmup_saved = True

    def predict(self, image: np.ndarray) -> Prediction:
        if not self._warmup_done():
            self.stad.warmup(image)
            self._warmup_counter += 1
            if self._warmup_counter == self._warmup_window_size:
                self._check_warmup_saved(Path("save/state/path"))
            return Prediction(image, False)

        heatmap, defects_mask, is_anomalous = self.stad.predict(image)

        if not is_anomalous:
            return Prediction(image, False)

        out_img, is_anomalous = self._draw_anomaly(defects_mask, image)
        return Prediction(out_img, is_anomalous)
