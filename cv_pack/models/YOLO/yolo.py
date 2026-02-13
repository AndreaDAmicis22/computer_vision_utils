import cv2
import numpy as np
import onnxruntime as ort


class YOLO_ONNX:
    """YOLO ONNX inference pipeline (axis-aligned bounding boxes)."""

    def __init__(self, model_path: str, input_size: int = 640, use_cuda: bool = False) -> None:
        """
        Args:
            model_path: Path to ONNX model.
            input_size: Square input size for the model.
            use_cuda: Whether to use CUDA execution provider.
        """
        self.input_size = input_size

        available_providers = ort.get_available_providers()
        if use_cuda and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("ONNX Runtime using CUDAExecutionProvider")  # noqa: T201
        elif not use_cuda and "OpenVINOExecutionProvider" in available_providers:
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
            print("ONNX Runtime using OpenVINOExecutionProvider")  # noqa: T201
        else:
            providers = ["CPUExecutionProvider"]
            print("ONNX Runtime using CPUExecutionProvider")  # noqa: T201

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)

        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (nw, nh))

        pad_x = (self.input_size - nw) // 2
        pad_y = (self.input_size - nh) // 2

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized

        blob = canvas.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]  # (1, 3, H, W)

        return blob, scale, pad_x, pad_y

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """
        Run inference on an image.
        Returns:
            detections: (N, 6) -> [x, y, w, h, score, class_id]
        """
        blob, scale, pad_x, pad_y = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        return outputs[0], scale, pad_x, pad_y

    def postprocess(
        self, detections: np.ndarray, scale: float, pad_x: int, pad_y: int, score_thr: float = 0.5
    ) -> list[dict]:
        """
        Convert raw detections to bounding boxes in original image coordinates.
        """
        results = []
        for det in detections:
            x, y, w, h, score, cls = det
            if score < score_thr:
                continue

            # Undo padding
            x -= pad_x
            y -= pad_y

            # Undo scaling
            x /= scale
            y /= scale
            w /= scale
            h /= scale

            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            results.append({"class_id": int(cls), "score": float(score), "bbox": [x1, y1, x2, y2]})

        return results

    @staticmethod
    def visualize(image: np.ndarray, detections: list[dict], class_names: list[str] | None = None) -> np.ndarray:
        """
        Draw axis-aligned bounding boxes.
        """
        vis = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            cls_id = det["class_id"]

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label = f"{class_names[cls_id]} {score:.2f}" if class_names else f"{cls_id} {score:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return vis
