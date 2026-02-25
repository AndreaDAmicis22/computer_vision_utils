from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class Providers(Enum):
    CPU = ("CPUExecutionProvider", {})
    OPENVINO_CPU = (
        ("OpenVINOExecutionProvider", {"device_type": "CPU", "precision": "FP32"}),
        ("CPUExecutionProvider", {}),
    )
    OPENVINO_GPU = (
        ("OpenVINOExecutionProvider", {"device_type": "GPU", "precision": "FP16"}),
        ("CPUExecutionProvider", {}),
    )


class FeatureExtractor:
    def __init__(self, onnx_model_path: Path, providers: Providers):
        self._session = ort.InferenceSession(
            onnx_model_path, providers=list(providers.value)
        )
        self.input_name = self._session.get_inputs()[0].name
        self.output_names = [o.name for o in self._session.get_outputs()]

    @staticmethod
    def preprocess(image: np.ndarray, new_size: int) -> np.ndarray:
        """Resize image to (new_size,new_size), convert to float32, normalize and reshape to (1, 3, H, W).

        Parameters
        ----------
        image : np.ndarray
            Image to preprocess
        new_size : int
            New size for resizing. It depends on the model's input

        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        image = cv2.resize(image, (new_size, new_size), cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5  # normalize
        image = np.transpose(image, (2, 0, 1))  # CHW
        return image[np.newaxis, ...]  # (1, 3, H, W)

    def extract_features(
        self, image: np.ndarray, target_image_size: int
    ) -> tuple[np.ndarray]:
        x = self.preprocess(image, new_size=target_image_size)
        cls, patches = self._session.run(
            self.output_names,
            {self.input_name: x},
        )
        return patches[0], cls[0]
