import os
import random
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

MODEL_SIZE_MAP = {
    "h": "vit_huge_plus_patch16_dinov3.lvd1689m",
    "l": "vit_large_patch16_dinov3.lvd1689m",
    "b": "vit_base_patch16_dinov3.lvd1689m",
    "sp": "vit_small_plus_patch16_dinov3.lvd1689m",
    "s": "vit_small_patch16_dinov3.lvd1689m",
}


class SlidingWindowAnomalyDetector:
    def __init__(
        self,
        model_size="l",
        save_dir="/workspace/data/VTR002/SLIDING_DATASET/anomaly_visualizations",
        warmup_window_size=15,
        inference_window_size=30,
        ramp_start=16,
        ramp_end=50,
        start_gamma=0.3,
        max_sane_visualizations=30,
        area_threshold=200,
        global_area_threshold=200,
        anomaly_min=100.0,
        anomaly_max=1300.0,
        target_img_size=512,
        device=None,
    ):
        if model_size not in MODEL_SIZE_MAP:
            msg = f"model_size '{model_size}' not valid. Possible values: {list(MODEL_SIZE_MAP.keys())}"
            raise ValueError(msg)

        self.model_name = MODEL_SIZE_MAP[model_size]
        self.model_size = model_size
        self.warmup_window_size = warmup_window_size
        self.inference_window_size = inference_window_size
        self.ramp_start = ramp_start
        self.ramp_end = ramp_end
        self.start_gamma = start_gamma
        self.max_sane_visualizations = max_sane_visualizations
        self.area_threshold = area_threshold
        self.global_area_threshold = global_area_threshold
        self.anomaly_min = anomaly_min
        self.anomaly_max = anomaly_max
        self.target_img_size = target_img_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Feature extractor
        self.extractor = self._build_extractor()

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
    def _preprocess(self, img_pil):
        size = (self.target_img_size, self.target_img_size)
        transform = T.Compose(
            [
                T.Resize(size),
                T.CenterCrop(size),
                T.ToTensor(),
                T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )
        return transform(img_pil).unsqueeze(0)

    # --- Calcolo media e inversa covarianza torch ---
    def _compute_mean_cov_inv(self, features, eps=1e-2):
        features = torch.tensor(features, dtype=torch.float32, device=self.device)
        mean = torch.mean(features, dim=0)
        diffs = features - mean
        cov = torch.matmul(diffs.T, diffs) / (features.shape[0] - 1)
        cov += eps * torch.eye(cov.shape[0], device=self.device)
        cov_inv = torch.linalg.inv(cov)
        return mean, cov_inv

    # --- Distanza Mahalanobis vettoriale ---
    def _mahalanobis_distance(self, vec, mean, cov_inv):
        diff = vec - mean
        left = torch.matmul(diff, cov_inv)
        d_sq = torch.sum(left * diff, dim=1)
        return torch.sqrt(d_sq)

    # --- Salva plot ---
    def _save_visualization(self, idx, img_pil, anomaly_map_norm, binary_mask, label, filename):
        _fig, axs = plt.subplots(1, 3, figsize=(11, 5))
        axs[0].imshow(img_pil)
        axs[0].set_title(f"Image {idx} ({label})")
        axs[0].axis("off")

        if anomaly_map_norm is not None:
            anomaly_map_resized = cv2.resize(
                anomaly_map_norm.cpu().numpy(),
                img_pil.size,
                interpolation=cv2.INTER_LINEAR,
            )
            axs[1].imshow(anomaly_map_resized, cmap="inferno", vmin=0.0, vmax=1.0)
            axs[1].set_title("Anomaly Map")
            axs[1].axis("off")
            plt.colorbar(
                axs[1].imshow(anomaly_map_resized, cmap="inferno", vmin=0.0, vmax=1.0),
                ax=axs[1],
            )

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
    # Loop Sliding Window
    # --------------------------
    def run(self, image_folder, bad_indices_path):
        saved_sane_vis_count = 0

        with open(bad_indices_path) as f:
            bad_indices_true = {int(line.strip()) for line in f if line.strip().isdigit()}

        img_filenames = sorted(os.listdir(image_folder), key=lambda x: int(os.path.splitext(x)[0]))
        img_paths = [os.path.join(image_folder, f) for f in img_filenames]

        memory_cls_features = []
        memory_patch_features = []
        good_cls_distances = []

        detected_bad_indices = []
        all_distances = []
        all_labels = []
        y_pred = []

        start_time = time.time()

        for idx, img_path in enumerate(img_paths, start=1):
            img_pil = Image.open(img_path).convert("RGB")

            img_np = np.array(img_pil)  # (H, W, C)
            img_pil = Image.fromarray(img_np)

            x = self._preprocess(img_pil).to(self.device)

            patch_tokens, cls_token = self._extract_features(x)
            patch_tokens = patch_tokens.squeeze(0)  # (N, D)
            cls_token = cls_token.squeeze(0)  # (D,1)

            # Calcolate distance if memory contains at least 2 instances
            if len(memory_cls_features) >= 2:
                # CLS
                cls_stack = torch.stack(memory_cls_features)
                mean_cls, covinv_cls = self._compute_mean_cov_inv(cls_stack)
                dist_cls = self._mahalanobis_distance(cls_token.unsqueeze(0), mean_cls, covinv_cls).item()

                # PATCH
                patch_stack = torch.cat(memory_patch_features, dim=0)
                mean_patch, covinv_patch = self._compute_mean_cov_inv(patch_stack)
                num_real_patches = (self.target_img_size // 16) ** 2  # 1024
                patch_tokens_trimmed = patch_tokens[-num_real_patches:, :]
                diff = patch_tokens_trimmed - mean_patch.unsqueeze(0)
                patch_maha_scores = torch.einsum(
                    "nd,df,nf->n", diff, covinv_patch, diff
                )  # Mahalanobis distance per patch: diffᵀ Σ⁻¹ diff
                grid_size = int(np.sqrt(num_real_patches))
                anomaly_map = patch_maha_scores.reshape(grid_size, grid_size)

                # Anomaly map reshape
                anomaly_map_norm = (anomaly_map - self.anomaly_min) / (self.anomaly_max - self.anomaly_min)
                anomaly_map_norm = torch.clamp(anomaly_map_norm, 0.0, 1.0)

                H, W = x.shape[2], x.shape[3]
                anomaly_map_up = F.interpolate(
                    anomaly_map_norm.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
                ).squeeze()

                # Connected components
                binary_mask_np = (anomaly_map_up > 0.6).cpu().numpy().astype(np.uint8)
                _num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask_np, connectivity=8)

                filtered_mask = np.zeros_like(binary_mask_np)
                total_anomalous_area = 0
                for label, stat in enumerate(stats[1:], start=1):
                    x_, y_, w, h, _area = stat
                    box_area = w * h
                    if box_area >= self.area_threshold:
                        filtered_mask[labels == label] = 1
                        total_anomalous_area += box_area
                binary_mask = torch.from_numpy(filtered_mask)
            else:
                dist_cls = 0.0
                total_anomalous_area = 0.0
                anomaly_map_norm = None
                anomaly_map_up = None
                binary_mask = None

            # Threshold dinamico
            if len(good_cls_distances) > 1:
                # std_cls = np.std(good_cls_distances)
                p70 = np.percentile(good_cls_distances, 70)

                if idx <= self.ramp_start:
                    gamma = self.start_gamma
                elif idx >= self.ramp_end:
                    gamma = 1.0
                else:
                    gamma = self.start_gamma + (1.0 - self.start_gamma) * (idx - self.ramp_start) / (
                        self.ramp_end - self.ramp_start
                    )

                # threshold = gamma * p99 + std_cls
                threshold = gamma * p70
            else:
                threshold = float("inf")
                gamma = 1.0

            print(  # noqa: T201
                f"Image {idx} - Mahalanobis distance CLS = {dist_cls:.3f} - Threshold = {threshold:.3f} - Anomalous area: {total_anomalous_area:.4f} - Ramping up gamma: {gamma:.2f}"
            )

            if idx > self.warmup_window_size:
                is_anomaly = dist_cls > threshold and total_anomalous_area > self.global_area_threshold
                all_distances.append(dist_cls)
                all_labels.append(1 if idx in bad_indices_true else 0)
                y_pred.append(1 if is_anomaly else 0)

                if is_anomaly:
                    print(f"Image {idx} - ANOMALY DETECTED")  # noqa: T201
                    detected_bad_indices.append(idx)

                    if idx not in bad_indices_true:
                        self._save_visualization(
                            idx, img_pil, anomaly_map_norm, binary_mask, "False Positive", "false_positive"
                        )

                    self._save_visualization(idx, img_pil, anomaly_map_norm, binary_mask, "Anomaly", "anomaly")

                else:
                    if idx in bad_indices_true:
                        self._save_visualization(
                            idx, img_pil, anomaly_map_norm, binary_mask, "False Negative", "false_negative"
                        )

                    if len(memory_cls_features) >= self.inference_window_size:
                        memory_cls_features.pop(0)
                        memory_patch_features.pop(0)
                        good_cls_distances.pop(0)

                    memory_cls_features.append(cls_token)
                    memory_patch_features.append(patch_tokens)
                    good_cls_distances.append(dist_cls)

                    if saved_sane_vis_count < self.max_sane_visualizations and idx > 100 and random.random() < 0.1:
                        self._save_visualization(idx, img_pil, anomaly_map_up, binary_mask, "Healthy", "healthy")
                        saved_sane_vis_count += 1
            else:
                print(f"Image {idx} - Added to memory (warm-up)")  # noqa: T201
                memory_cls_features.append(cls_token)
                memory_patch_features.append(patch_tokens)
                good_cls_distances.append(dist_cls)

        # --- model evaluation ---
        total_time = time.time() - start_time
        avg_time_per_image = total_time / len(img_paths)
        print(f"\nProcessed {len(img_paths)} images in {total_time:.2f} seconds.")  # noqa: T201
        print(f"Average time per image: {avg_time_per_image:.4f} seconds")  # noqa: T201

        print(f"\nTrue bad indices: {sorted(bad_indices_true)}")  # noqa: T201
        print(f"Detected bad indices: {sorted(detected_bad_indices)}")  # noqa: T201

        if len(all_labels) > 0:
            f1 = f1_score(all_labels, y_pred)
            precision = precision_score(all_labels, y_pred)
            recall = recall_score(all_labels, y_pred)
            try:
                auroc = roc_auc_score(all_labels, all_distances)
            except ValueError:
                auroc = float("nan")
        else:
            f1 = precision = recall = auroc = float("nan")

        print("\n--- Performance Metrics ---")  # noqa: T201
        print(f"F1 Score     : {f1:.2f}")  # noqa: T201
        print(f"Precision    : {precision:.2f}")  # noqa: T201
        print(f"Recall       : {recall:.2f}")  # noqa: T201
        print(f"AUROC        : {auroc:.2f}")  # noqa: T201

        if len(all_labels) > 0:
            cm = confusion_matrix(all_labels, y_pred)
            plt.figure(figsize=(4, 4))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"],
            )
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, "confusion_matrix_sliding.png"), dpi=300)
            plt.close()

        return detected_bad_indices, bad_indices_true, f1, auroc


"""
TO USE WITH THE FOLLOWING CODE IN ANOTHER NOTEBOOK

from sliding_AD_dinov3 import SlidingWindowAnomalyDetector

detector = SlidingWindowAnomalyDetector(save_dir="/workspace/data/SLIDING_DATASET/anomaly_visualizations_dinov3_modified2")

detected_bad, true_bad, f1, auroc = detector.run_inference(
    image_folder="/workspace/data/SLIDING_DATASET/mixed",
    bad_indices_path="/workspace/data/SLIDING_DATASET/bad_indices_last.txt"
)
"""
