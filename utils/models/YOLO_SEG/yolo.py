import cv2
import numpy as np
import onnxruntime as ort


class YOLO_SEG_ONNX:
    """YOLO SEG ONNX inference pipeline (bbox + mask)."""

    def __init__(self, model_path: str, input_size: int = 640, use_cuda: bool = False) -> None:
        self.input_size = input_size

        available_providers = ort.get_available_providers()
        if use_cuda and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif not use_cuda and "OpenVINOExecutionProvider" in available_providers:
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name  # Assumiamo output unico

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
        blob = np.transpose(blob, (2, 0, 1))[None]

        return blob, scale, pad_x, pad_y

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """
        Run inference on an image.
        Returns:
            detections_with_masks: (N, 6 + mask_dim)
        """
        blob, scale, pad_x, pad_y = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: blob})
        return outputs[0], scale, pad_x, pad_y

    def postprocess(
        self, outputs: np.ndarray, scale: float, pad_x: int, pad_y: int, score_thr: float = 0.5
    ) -> list[dict]:
        """
        Convert raw outputs to bounding boxes and masks in original image coordinates.
        Assumes outputs shape: (N, 6 + mask_dim) -> [x, y, w, h, score, class_id, mask...]
        """
        results = []
        for det in outputs:
            x, y, w, h, score, cls, *mask_flat = det
            if score < score_thr:
                continue

            # Undo padding and scaling
            x = (x - pad_x) / scale
            y = (y - pad_y) / scale
            w /= scale
            h /= scale

            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)

            # Reshape mask to (mask_h, mask_w) if needed
            if mask_flat:
                mask = np.array(mask_flat, dtype=np.float32)
                mask_h = mask_w = int(np.sqrt(len(mask_flat)))  # assume square mask
                mask = mask.reshape(mask_h, mask_w)
                # Resize mask to bbox size
                mask_resized = cv2.resize(mask, (x2 - x1, y2 - y1))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
            else:
                mask_binary = None

            results.append({"class_id": int(cls), "score": float(score), "bbox": [x1, y1, x2, y2], "mask": mask_binary})

        return results

    @staticmethod
    def visualize(image: np.ndarray, detections: list[dict], class_names: list[str] | None = None) -> np.ndarray:
        vis = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            score = det["score"]
            cls_id = det["class_id"]
            mask = det.get("mask")

            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_names[cls_id]} {score:.2f}" if class_names else f"{cls_id} {score:.2f}"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if mask is not None:
                # Color overlay for mask
                colored_mask = np.zeros_like(vis, dtype=np.uint8)
                colored_mask[y1:y2, x1:x2, 1] = mask * 255
                vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.5, 0)

        return vis
