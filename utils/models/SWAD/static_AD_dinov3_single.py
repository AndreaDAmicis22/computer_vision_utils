import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


class DinoV3AnomalyDetector:
    def __init__(
        self,
        save_dir,
        model_name="vit_base_patch16_dinov3.lvd1689m",
        device=None,
        area_threshold=200,
        anomaly_min=100.0,
        anomaly_max=1300.0,
        cls_threshold=10,
        tot_area_threshold=500,
        blob_area_threshold=20,
        target_size=464,
    ):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model_name = model_name
        self.area_threshold = area_threshold
        self.anomaly_min = anomaly_min
        self.anomaly_max = anomaly_max
        self.cls_threshold = cls_threshold
        self.tot_area_threshold = tot_area_threshold
        self.blob_area_threshold = blob_area_threshold
        self.target_size = target_size

        # Feature extractor
        self.extractor = self._build_extractor()

        # Variabili di training
        self.mean_cls = None
        self.covinv_cls = None
        self.mean_patch = None
        self.covinv_patch = None

    # --------------------------
    # Feature extractor
    # --------------------------
    def _build_extractor(self):
        model = timm.create_model(self.model_name, pretrained=True)
        model.eval()
        return model.to(self.device)

    def _extract_features(self, x):
        with torch.no_grad():
            features = self.extractor.forward_features(x)
            patch_tokens = features[:, 1:, :]  # senza CLS
            cls_token = features[:, 0, :]
        return patch_tokens, cls_token

    # --------------------------
    # Utils
    # --------------------------
    def _preprocess(self, img_pil, size=(512, 512)):
        transform = T.Compose(
            [
                T.Resize(size),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        return transform(img_pil).unsqueeze(0)

    def _compute_mean_cov_inv(self, features, eps=1e-2):
        features = torch.tensor(features, dtype=torch.float32, device=self.device)
        mean = torch.mean(features, dim=0)
        diffs = features - mean
        cov = torch.matmul(diffs.T, diffs) / (features.shape[0] - 1)
        cov += eps * torch.eye(cov.shape[0], device=self.device)
        cov_inv = torch.linalg.inv(cov)
        return mean, cov_inv

    def _mahalanobis_distance(self, vec, mean, cov_inv):
        diff = vec - mean
        left = torch.matmul(diff, cov_inv)
        d_sq = torch.sum(left * diff, dim=1)
        return torch.sqrt(d_sq)

    def _save_visualization(self, idx, img_pil, anomaly_map_norm, binary_mask, label, filename):
        _fig, axs = plt.subplots(1, 3, figsize=(11, 5))
        axs[0].imshow(img_pil)
        axs[0].set_title(f"Image {idx} ({label})")
        axs[0].axis("off")

        if anomaly_map_norm is not None:
            anomaly_map_resized = cv2.resize(
                anomaly_map_norm.cpu().numpy(), img_pil.size, interpolation=cv2.INTER_LINEAR
            )
            axs[1].imshow(anomaly_map_resized, cmap="inferno", vmin=0.0, vmax=1.0)
            axs[1].set_title("Anomaly Map")
            axs[1].axis("off")
            plt.colorbar(axs[1].imshow(anomaly_map_resized, cmap="inferno", vmin=0.0, vmax=1.0), ax=axs[1])

            axs[2].imshow(binary_mask.cpu().numpy(), cmap="copper")
            axs[2].set_title("Binary Mask")
            axs[2].axis("off")
        else:
            axs[1].text(0.5, 0.5, "No anomaly map", ha="center")
            axs[1].axis("off")
            axs[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(self.save_dir, f"{filename}_{idx}.png")
        plt.savefig(save_path)
        plt.close()

    # --------------------------
    # Training
    # --------------------------
    def fit(self, training_dir):
        cls_features = []
        patch_features = []
        good_cls_distances = []

        img_paths = sorted(
            [
                os.path.join(training_dir, f)
                for f in os.listdir(training_dir)
                if f.lower().endswith((".jpg", ".png", ".bmp"))
            ]
        )

        for img_path in img_paths:
            img_pil = Image.open(img_path).convert("RGB")
            x = self._preprocess(img_pil).to(self.device)

            patch_tokens, cls_token = self._extract_features(x)
            cls_features.append(cls_token.squeeze(0))
            patch_features.append(patch_tokens.squeeze(0))

        print(f"[Training] Estratte feature da {len(img_paths)} immagini sane")

        # Media e covarianza
        cls_stack = torch.stack(cls_features)
        patch_stack = torch.cat(patch_features, dim=0)

        self.mean_cls, self.covinv_cls = self._compute_mean_cov_inv(cls_stack)
        self.mean_patch, self.covinv_patch = self._compute_mean_cov_inv(patch_stack)

        # Threshold CLS
        for cls_token in cls_features:
            dist_cls = self._mahalanobis_distance(cls_token.unsqueeze(0), self.mean_cls, self.covinv_cls).item()
            good_cls_distances.append(dist_cls)

        std_cls = np.std(good_cls_distances)
        p99 = np.percentile(good_cls_distances, 99)
        self.cls_threshold = 1.0 * p99 + std_cls

        print(f"[Training] STATIC_CLS_THRESHOLD = {self.cls_threshold:.4f}")
        return self.cls_threshold, self.mean_patch, self.covinv_patch

    # --------------------------
    # Inference
    # --------------------------
    def predict(self, frames, cls_threshold):
        cls_threshold = self.cls_threshold if cls_threshold is None else cls_threshold

        # Assumiamo che frames sia una lista di PIL Images o tensor già preprocessati
        img = frames[0] if isinstance(frames[0], Image.Image) else frames[0]

        # Preprocess
        x = self._preprocess(img).to(self.device) if isinstance(img, Image.Image) else img.to(self.device)

        # Estrazione feature
        patch_tokens, cls_token = self._extract_features(x)
        patch_tokens = patch_tokens.squeeze(0)
        cls_token = cls_token.squeeze(0)

        # Distanza CLS
        dist_cls = self._mahalanobis_distance(cls_token.unsqueeze(0), self.mean_cls, self.covinv_cls).item()

        # Patch anomaly map
        num_real_patches = (512 // 16) ** 2
        patch_tokens_trimmed = patch_tokens[-num_real_patches:, :]
        diff = patch_tokens_trimmed - self.mean_patch.unsqueeze(0)
        patch_maha_scores = torch.einsum("nd,df,nf->n", diff, self.covinv_patch, diff)
        grid_size = int(np.sqrt(num_real_patches))
        anomaly_map = patch_maha_scores.reshape(grid_size, grid_size)

        # Normalizzazione globale
        anomaly_map_norm = (anomaly_map - self.anomaly_min) / (self.anomaly_max - self.anomaly_min)
        anomaly_map_norm = torch.clamp(anomaly_map_norm, 0.0, 1.0)

        H, W = x.shape[2], x.shape[3]
        anomaly_map_up = F.interpolate(
            anomaly_map_norm.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze()

        # Filtering per area
        binary_mask_np = (anomaly_map_up > 0.6).cpu().numpy().astype(np.uint8)
        _num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_np, connectivity=8)
        filtered_mask = np.zeros_like(binary_mask_np)
        total_anomalous_area = 0
        for label, stat in enumerate(stats[1:], start=1):
            _x0, _y0, w, h, _area = stat
            box_area = w * h
            if box_area >= self.area_threshold:
                filtered_mask[labels == label] = 1
                total_anomalous_area += box_area

        binary_mask = torch.from_numpy(filtered_mask)

        # Decisione di anomalia
        fault = dist_cls > cls_threshold and total_anomalous_area > self.area_threshold

        # Probabilità di essere anomalo come discard_value
        image_area = H * W
        discard_value = np.clip(float(total_anomalous_area) / float(image_area), 0, 1)

        # Visualizzazione (draw)
        # Visualizzazione (draw) salvata
        save_dir = "/workspace/data/SPA006/output_fit"
        os.makedirs(save_dir, exist_ok=True)

        _fig, axs = plt.subplots(1, 3, figsize=(11, 5))
        axs[0].imshow(img if isinstance(img, Image.Image) else img.permute(1, 2, 0).cpu())
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        anomaly_map_resized = cv2.resize(anomaly_map_norm.cpu().numpy(), (W, H), interpolation=cv2.INTER_LINEAR)
        im1 = axs[1].imshow(anomaly_map_resized, cmap="inferno", vmin=0.0, vmax=1.0)
        axs[1].set_title("Anomaly Map")
        axs[1].axis("off")
        plt.colorbar(im1, ax=axs[1])

        axs[2].imshow(binary_mask.cpu().numpy(), cmap="copper")
        axs[2].set_title("Binary Mask")
        axs[2].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"prediction_{np.random.randint(1e6)}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        # Return con anomaly map come draw
        return {"discard": discard_value, "draw": anomaly_map_resized, "fault": fault}

    def predict_chandra(self, frame, cls_threshold):
        # convert directly to RGB tensor
        if isinstance(frame, Image.Image):
            x = self._preprocess(frame).to(self.device)
        elif isinstance(frame, np.ndarray):
            if frame.ndim == 2:  # grayscale
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 1:
                frame = np.concatenate([frame] * 3, axis=-1)
            elif frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x = self._preprocess(Image.fromarray(frame.astype(np.uint8))).to(self.device)
        elif isinstance(frame, torch.Tensor):
            x = frame.to(self.device) if frame.ndim == 4 else frame.unsqueeze(0).to(self.device)
        else:
            msg = f"Unsupported frame type: {type(frame)}"
            raise TypeError(msg)

        # Feature extraction
        with torch.no_grad():
            # a = time.time()
            patch_tokens, cls_token = self._extract_features(x)
            # logger.info(f"Feats extracted in: {(time.time() - a):.2f} s")

        patch_tokens = patch_tokens.squeeze(0)
        cls_token = cls_token.squeeze(0)

        # CLS distance
        dist_cls = self._mahalanobis_distance(cls_token.unsqueeze(0), self.mean_cls, self.covinv_cls).item()

        # Patch anomaly map
        num_real_patches = (self.target_size // 16) ** 2
        patch_tokens_trimmed = patch_tokens[-num_real_patches:, :]
        diff = patch_tokens_trimmed - self.mean_patch.unsqueeze(0)
        patch_maha_scores = torch.einsum("nd,df,nf->n", diff, self.covinv_patch, diff)
        grid_size = int(np.sqrt(num_real_patches))
        anomaly_map = patch_maha_scores.view(grid_size, grid_size)

        # Normalization and upsampling
        anomaly_map_norm = torch.clamp(
            (anomaly_map - self.anomaly_min) / (self.anomaly_max - self.anomaly_min),
            0.0,
            1.0,
        )
        H, W = x.shape[2], x.shape[3]
        anomaly_map_up = F.interpolate(
            anomaly_map_norm[None, None, ...],
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        # Filtering area
        binary_mask = (anomaly_map_up > 0.6).cpu().numpy().astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

        filtered_mask = np.zeros_like(binary_mask)
        total_anomalous_area = 0

        for i, (_x0, _y0, w, h, _area) in enumerate(stats[1:], start=1):  # skip background
            blob_area = w * h
            if blob_area >= self.blob_area_threshold:
                total_anomalous_area += blob_area
                filtered_mask[labels == i] = 1

        # Decision
        is_anomaly = dist_cls > cls_threshold and total_anomalous_area > self.tot_area_threshold

        anomaly_map_resized = cv2.resize(
            anomaly_map_norm.cpu().numpy(),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        return {"fault": dist_cls, "draw": anomaly_map_resized, "discard": is_anomaly}
