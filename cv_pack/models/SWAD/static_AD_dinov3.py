import os
import cv2
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from PIL import Image


class DinoV3AnomalyDetector:
    def __init__(self, 
                 save_dir,
                 model_name="vit_large_patch16_dinov3.lvd1689m",
                 device=None,
                 area_threshold=200,
                 anomaly_min=100.0,
                 anomaly_max=1300.0):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.area_threshold = area_threshold
        self.anomaly_min = anomaly_min
        self.anomaly_max = anomaly_max

        # Feature extractor
        self.extractor = self._build_extractor()

        # Variabili di training
        self.mean_cls = None
        self.covinv_cls = None
        self.mean_patch = None
        self.covinv_patch = None
        self.cls_threshold = None

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
        transform = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),
        ])
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
        fig, axs = plt.subplots(1, 3, figsize=(11, 5))
        axs[0].imshow(img_pil)
        axs[0].set_title(f"Image {idx} ({label})")
        axs[0].axis('off')

        if anomaly_map_norm is not None:
            anomaly_map_resized = cv2.resize(
                anomaly_map_norm.cpu().numpy(),
                img_pil.size,
                interpolation=cv2.INTER_LINEAR
            )
            axs[1].imshow(anomaly_map_resized, cmap='inferno', vmin=0.0, vmax=1.0)
            axs[1].set_title("Anomaly Map")
            axs[1].axis('off')
            plt.colorbar(axs[1].imshow(anomaly_map_resized, cmap='inferno', vmin=0.0, vmax=1.0), ax=axs[1])

            axs[2].imshow(binary_mask.cpu().numpy(), cmap='copper')
            axs[2].set_title("Binary Mask")
            axs[2].axis('off')
        else:
            axs[1].text(0.5, 0.5, "No anomaly map", ha='center')
            axs[1].axis('off')
            axs[2].axis('off')

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
            [os.path.join(training_dir, f) for f in os.listdir(training_dir)
             if f.lower().endswith((".jpg", ".png", ".bmp"))]
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

    # --------------------------
    # Inference
    # --------------------------
    def predict(self, image_dir, bad_indices_path):
        with open(bad_indices_path, "r") as f:
            bad_indices_true = set(int(line.strip()) for line in f if line.strip().isdigit())

        img_filenames = sorted(os.listdir(image_dir), key=lambda x: int(os.path.splitext(x)[0]))
        img_paths = [os.path.join(image_dir, f) for f in img_filenames]

        detected_bad_indices, all_distances, all_labels, y_pred = [], [], [], []

        for idx, img_path in enumerate(img_paths, start=1):
            img_pil = Image.open(img_path).convert("RGB")
            x = self._preprocess(img_pil).to(self.device)

            patch_tokens, cls_token = self._extract_features(x)
            patch_tokens = patch_tokens.squeeze(0)
            cls_token = cls_token.squeeze(0)

            # Distanza CLS
            dist_cls = self._mahalanobis_distance(cls_token.unsqueeze(0), self.mean_cls, self.covinv_cls).item()

            # Patch anomaly map
            num_real_patches = (512 // 16) ** 2
            patch_tokens_trimmed = patch_tokens[-num_real_patches:, :]

            diff = patch_tokens_trimmed - self.mean_patch.unsqueeze(0)
            patch_maha_scores = torch.einsum('nd,df,nf->n', diff, self.covinv_patch, diff)
            grid_size = int(np.sqrt(num_real_patches))
            anomaly_map = patch_maha_scores.reshape(grid_size, grid_size)

            anomaly_map_norm = (anomaly_map - self.anomaly_min) / (self.anomaly_max - self.anomaly_min)
            anomaly_map_norm = torch.clamp(anomaly_map_norm, 0.0, 1.0)

            H, W = x.shape[2], x.shape[3]
            anomaly_map_up = F.interpolate(
                anomaly_map_norm.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False
            ).squeeze()

            # Filtering per area
            binary_mask_np = (anomaly_map_up > 0.6).cpu().numpy().astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_np, connectivity=8)
            filtered_mask = np.zeros_like(binary_mask_np)
            total_anomalous_area = 0
            for label, stat in enumerate(stats[1:], start=1):
                x0, y0, w, h, area = stat
                box_area = w * h
                if box_area >= self.area_threshold:
                    filtered_mask[labels == label] = 1
                    total_anomalous_area += box_area

            binary_mask = torch.from_numpy(filtered_mask)

            # Decisione
            is_anomaly = dist_cls > self.cls_threshold and total_anomalous_area > self.area_threshold

            all_distances.append(dist_cls)
            all_labels.append(1 if idx in bad_indices_true else 0)
            y_pred.append(1 if is_anomaly else 0)

            if is_anomaly:
                detected_bad_indices.append(idx)
                if idx not in bad_indices_true:
                    self._save_visualization(idx, img_pil, anomaly_map_norm, binary_mask, "False Positive", "false_positive")
                self._save_visualization(idx, img_pil, anomaly_map_norm, binary_mask, "Anomaly", "anomaly")
            else:
                if idx in bad_indices_true:
                    self._save_visualization(idx, img_pil, anomaly_map_norm, binary_mask, "False Negative", "false_negative")
                else:
                    if random.random() < 0.1:
                        self._save_visualization(idx, img_pil, anomaly_map_up, binary_mask, "Healthy", "healthy")

            print(f"Image {idx} - CLS Mahalanobis: {dist_cls:.3f} - Area: {total_anomalous_area} - Anomaly: {is_anomaly}")

        # Metriche
        if len(all_labels) > 0:
            f1 = f1_score(all_labels, y_pred)
            precision = precision_score(all_labels, y_pred)
            recall = recall_score(all_labels, y_pred)
            try:
                auroc = roc_auc_score(all_labels, all_distances)
            except ValueError:
                auroc = float('nan')
        else:
            f1 = precision = recall = auroc = float('nan')

        print(f"\n--- Performance Metrics ---")
        print(f"F1 Score     : {f1:.2f}")
        print(f"Precision    : {precision:.2f}")
        print(f"Recall       : {recall:.2f}")
        print(f"AUROC        : {auroc:.2f}")

        cm = confusion_matrix(all_labels, y_pred)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Anomaly"], yticklabels=["Normal", "Anomaly"])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "confusion_matrix.png"), dpi=300)
        plt.close()

        return detected_bad_indices, bad_indices_true, f1, auroc



"""
TO USE WITH THE FOLLOWING CODE IN ANOTHER NOTEBOOK

from static_AD_dinov3 import DinoV3AnomalyDetector

base_path = '/workspace/data/SPA006/241002-acquisitions/test_all'
training_dir = os.path.join(base_path, 'training')
mixed_dir = os.path.join(base_path, 'mixed')
bad_indices_path = os.path.join(base_path, 'bad_indices.txt')
SAVE_DIR = os.path.join(base_path, 'anomaly_visualizations_training2')

detector = DinoV3AnomalyDetector(save_dir=SAVE_DIR)

# Training su immagini sane
detector.fit(training_dir)

# Inference su immagini miste
detected, true_bad, f1, auroc = detector.predict(mixed_dir, bad_indices_path)

"""