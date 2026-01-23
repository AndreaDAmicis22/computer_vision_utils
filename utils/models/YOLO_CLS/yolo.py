import cv2
import numpy as np
import onnxruntime as ort


class YOLO_CLS_ONNX:
    """YOLO Classification ONNX inference pipeline."""

    def __init__(
        self, model_path: str, input_size: int = 640, use_cuda: bool = False, class_labels: list[str] | None = None
    ) -> None:
        """
        Args:
            model_path: Path to ONNX model.
            input_size: Square input size for the model.
            use_cuda: Whether to use CUDA execution provider.
            class_labels: List of class labels for the model.
        """
        self.input_size = input_size
        self.class_labels = class_labels

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

    # -------------------------
    # Preprocessing (YOLO style)
    # -------------------------
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        img = cv2.resize(image, (self.input_size, self.input_size))
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))[None]  # (1,3,H,W)

    # -------------------------
    # Inference
    # -------------------------
    def infer(self, image: np.ndarray) -> np.ndarray:
        blob = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        return outputs[0]  # shape: (1, num_classes)

    # -------------------------
    # Postprocessing
    # -------------------------
    def postprocess(self, logits: np.ndarray):
        probs = softmax(logits[0])
        cls_id = int(np.argmax(probs))
        score = float(probs[cls_id])

        result = {"class_id": cls_id, "score": score, "probs": probs}
        if self.class_labels is not None:
            # Se le label sono fornite, aggiungi la label testuale
            result["label"] = self.class_labels[cls_id]
        return result

    # -------------------------
    # Full pipeline
    # -------------------------
    def predict(self, image: np.ndarray):
        logits = self.infer(image)
        return self.postprocess(logits)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()
