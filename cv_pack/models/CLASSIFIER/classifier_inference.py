import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torchvision import transforms


class ClassifierONNX:
    def __init__(
        self,
        onnx_model_path,
        device="cpu",
        img_size=256,
        classes=None,
    ):
        if classes is None:
            classes = ["Cracked", "Flipped", "Normal"]
        self.device = device
        self.img_size = img_size
        self.onnx_model_path = onnx_model_path

        providers = self._get_providers()
        self.ort_session = ort.InferenceSession(str(self.onnx_model_path), providers=providers)
        self.classes = classes
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _get_providers(self) -> list[tuple[str, dict]]:
        """Logica di selezione del provider con priorità a OpenVINO."""
        available = ort.get_available_providers()
        providers = []

        # 1. OpenVINO come prima scelta per la massima velocità su CPU Intel
        if "OpenVINOExecutionProvider" in available:
            # Per OpenVINO, si può specificare il device se necessario (es. {"device_type": "CPU_FP32"})
            providers.append("OpenVINOExecutionProvider")

        # 2. CUDA come seconda scelta (se presente)
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")

        # 3. CPU come fallback finale
        providers.append("CPUExecutionProvider")

        return providers

    def predict(self, image_np, confidence_threshold: float = 0.6):
        if not isinstance(image_np, np.ndarray):
            msg = "Input deve essere un numpy.ndarray"
            raise ValueError(msg)

        # BGR -> RGB
        if image_np.shape[2] == 3:
            image_np = image_np[..., ::-1]
        if image_np.dtype == np.uint8:
            image_np = image_np.astype(np.float32) / 255.0

        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        img_tensor = self.transform(image_pil).unsqueeze(0)

        ort_inputs = {"input": img_tensor.cpu().numpy()}
        ort_outs = self.ort_session.run(None, ort_inputs)

        # Softmax sui logit
        probs = torch.softmax(torch.from_numpy(ort_outs[0][0]), dim=0).numpy()
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])

        # Confidence check
        if confidence >= confidence_threshold:
            class_name = self.classes[class_idx]
        else:
            class_name = "Flipped"
            class_idx = self.classes.index("Flipped")
            confidence = float(probs[class_idx])

        return class_name, confidence
