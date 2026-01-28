import cv2
import numpy as np
import onnxruntime as ort


class YOLO_SEG_ONNX:
    def __init__(self, model_path, input_size=1024, use_cuda=False, conf_threshold=0.25, mask_threshold=0.5):
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.mask_threshold = mask_threshold

        available_providers = ort.get_available_providers()
        if use_cuda and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif not use_cuda and "OpenVINOExecutionProvider" in available_providers:
            providers = ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)

        # Estrazione info modello
        model_inputs = self.session.get_inputs()
        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape  # [batch, 3, height, width]

    def preprocess(self, img):
        """Resize e normalizzazione mantenendo l'aspect ratio (letterbox)."""
        h, w = img.shape[:2]
        scale = min(self.input_size / h, self.input_size / w)
        nh, nw = int(h * scale), int(w * scale)

        img_resized = cv2.resize(img, (nw, nh))
        # Creiamo un canvas nero quadrato (letterbox)
        input_img = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        input_img[:nh, :nw, :] = img_resized

        # Conversione HWC -> CHW e normalizzazione 0-1
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.ascontiguousarray(input_img).astype(np.float32) / 255.0
        return input_img[None, :, :, :], (h, w), scale

    def postprocess(self, preds, orig_shape, scale):
        p_boxes = preds[0][0]
        p_proto = preds[1][0]

        # Estraiamo i punteggi massimi per ogni box
        scores = np.max(p_boxes[:, 4:-32], axis=1)
        mask = scores > self.conf_threshold

        p_boxes = p_boxes[mask]
        scores = scores[mask]  # Teniamo solo gli score validi

        if len(p_boxes) == 0:
            return [], [], [], []

        boxes = p_boxes[:, :4]
        class_ids = np.argmax(p_boxes[:, 4:-32], axis=1)
        mask_coeffs = p_boxes[:, -32:]

        # Generazione maschere (logica del crop inclusa)
        proto_h, proto_w = p_proto.shape[1:]
        masks = (mask_coeffs @ p_proto.reshape(32, -1)).reshape(-1, proto_h, proto_w)
        masks = 1 / (1 + np.exp(-masks))

        full_masks = []
        upscaled_h, upscaled_w = int(orig_shape[0] * scale), int(orig_shape[1] * scale)

        for i, m in enumerate(masks):
            m = cv2.resize(m, (self.input_size, self.input_size))

            # Clipping e Cropping
            x1, y1, x2, y2 = np.clip(boxes[i], 0, self.input_size).astype(int)
            crop_mask = np.zeros_like(m)
            crop_mask[y1:y2, x1:x2] = 1
            m = m * crop_mask

            m = m[:upscaled_h, :upscaled_w]
            m = cv2.resize(m, (orig_shape[1], orig_shape[0]))
            full_masks.append(m > self.mask_threshold)

        boxes /= scale
        return boxes, scores, class_ids, np.array(full_masks)

    def predict(self, frame):
        blob, orig_shape, scale = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: blob})

        boxes, scores, class_ids, masks = self.postprocess(outputs, orig_shape, scale)

        annotated_img = frame.copy()

        # Calcoliamo dinamicamente la scala del font in base alla larghezza dell'immagine
        # In questo modo su immagini grandi il font scala automaticamente
        _h, w = frame.shape[:2]
        font_scale = max(0.7, w / 1500)
        thickness = max(2, int(w / 700))

        for _i, (box, score, class_id, m) in enumerate(zip(boxes, scores, class_ids, masks, strict=False)):
            x1, y1, x2, y2 = box.astype(int)

            # 1. Colore (BGR) e Disegno BBox/Maschera
            color = (255, 20, 0)  # Blu acceso
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, thickness)

            # Overlay maschera
            annotated_img[m] = annotated_img[m] * 0.5 + np.array(color) * 0.5

            # 2. Preparazione testo
            label = f"ID:{class_id} {score:.2f}"

            # Recuperiamo le dimensioni del testo con la nuova scala
            (tw, th), _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

            # Disegniamo lo sfondo del testo (leggermente più grande per dare margine)
            # Spostiamo il testo sopra la box, o dentro se non c'è spazio in alto
            ty1 = max(y1, th + 10)
            cv2.rectangle(
                annotated_img, (x1, ty1 - th - 10), (x1 + tw + 10, ty1), color, -1
            )  # -1 riempie il rettangolo

            # Scriviamo il testo in bianco con spessore maggiorato
            cv2.putText(
                annotated_img,
                label,
                (x1 + 5, ty1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated_img, masks
