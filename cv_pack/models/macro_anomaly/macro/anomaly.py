from typing import Sequence, Literal

import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .net import ResNetFeatureExtractor, GaussianBlur2d
from .utils import align, revert


class MacroDefectDetector(nn.Module):
    """Class for detection of macro defects."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        layers: list[str] = ["layer2", "layer3"],
        metric: Literal["cosine", "euclidean"] = "euclidean",
        sigma: float = 4.0,
    ):
        super().__init__()
        self.resnet = ResNetFeatureExtractor(model_name, model_path, layers)
        self.feature_pooler = torch.nn.AvgPool2d(3, 1, 1)
        self.blur = GaussianBlur2d(sigma)
        self.transforms = T.Compose([T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.metric = metric

        self.nominal_features = None
        self.background_masks = None

    def extract_features(self, x: torch.Tensor, split: bool = True) -> torch.Tensor:
        """Extract features from image.

        Args:
            x (torch.Tensor): Image tensor, shape `(c, h, w)`.
            split (bool): Whether to split the input image in half across its width and process it
                as a batch of 2 images. Default `True`.

        Returns:
            out (torch.Tensor): Extracted features, shape `(1, C, H, W)`.
        """
        if split:
            c, h = x.shape[:2]
            x = x.reshape(c, h, 2, -1).permute(2, 0, 1, 3)
            x = self.transforms(x)

            fmaps = self.resnet(x)
            pyramid = self.merge_features([self.feature_pooler(fmap) for fmap in fmaps])

            e, hf = pyramid.shape[1:3]
            pyramid = pyramid.permute(1, 2, 0, 3).reshape(e, hf, -1)
            pyramid = pyramid.unsqueeze(0)
        else:
            x = x.unsqueeze(0)
            x = self.transforms(x)

            fmaps = self.resnet(x)
            pyramid = self.merge_features([self.feature_pooler(fmap) for fmap in fmaps])
        return pyramid

    def merge_features(self, fmaps: list[torch.Tensor]) -> torch.Tensor:
        """Merge feature maps at different resolutions."""
        pyramid = fmaps[0]
        for fmap in fmaps[1:]:
            upscaled = F.interpolate(fmap, size=pyramid.shape[-2:], mode="bilinear")
            pyramid = torch.cat((pyramid, upscaled), 1)
        return pyramid

    def get_anomaly_map(
        self,
        tgt_features: torch.Tensor,
        ref_features: torch.Tensor,
        out_size: tuple[int, int],
    ) -> torch.Tensor:
        """Generate anomaly map from test and reference features."""
        if self.metric == "euclidean":
            tgt_map_norm = F.normalize(tgt_features, dim=-3)
            ref_map_norm = F.normalize(ref_features, dim=-3)
            am = 0.5 * torch.norm(tgt_map_norm - ref_map_norm, 2, -3)
        elif self.metric == "cosine":
            am = 1 - F.cosine_similarity(tgt_features, ref_features, dim=1)
        else:
            raise ValueError(f'Invalid metric {self.metric}. Available choices are "euclidean" and "cosine".')

        return F.interpolate(am.unsqueeze(1), out_size, mode="bilinear")

    @torch.no_grad()
    def calibrate(
        self,
        segments: torch.Tensor | Sequence[torch.Tensor],
        background: torch.Tensor,
        shift: int = 6,
        thresh: int = 70,
    ) -> None:
        """Extract features from all segments in a reference image and compute background mask.

        Args:
            segments (torch.Tensor | Sequence[torch.Tensor]): Slices making up the entirety of the object. Either
                a `torch.Tensor` of shape `(n, 3, h, w)` or a `Sequence` of `n` tensors, each of shape `(3, h, w)`.
            background (torch.Tensor): Image of background, shape `(3, h, w)`.
            shift (int): Horizontal shift required to align adjacent segments. Default `6`.
            thresh (int): Threshold used to segment foreground. Default `70`.
        """
        print("Calibrating system...")
        h, w = background.shape[-2:]
        self.nominal_features = torch.stack([self.extract_features(segment) for segment in segments])
        background_features = self.extract_features(background)

        fg_map = torch.cat([self.get_anomaly_map(fmap, background_features, (h, w)) for fmap in self.nominal_features])
        fg_map = (255 * fg_map).squeeze().detach().cpu().numpy().astype(np.uint8)

        fg_map = align(fg_map.reshape(-1, h, w), shift)
        mask = cv2.threshold(fg_map, thresh, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)

        fg_mask = np.zeros_like(fg_map)
        cv2.drawContours(fg_mask, [hull], 0, 255, -1)

        self.background_masks = torch.from_numpy(np.stack([fg == 0 for fg in revert(fg_mask, shift, h)])).to(
            background.device
        )
        print("System calibrated successfully.")

    @torch.no_grad()
    def predict(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        # TODO fix potential memory leak
        """Get anomaly map of given segment.

        Args:
            x (torch.Tensor): Image segment, shape `(1, 3, h, w)`.
            idx (int): Index of segment with respect to the entire image.

        Returns:
            amap (torch.Tensor): Anomaly map masked by foreground mask obtained in the calibration process.
                Shape `(1, 1, h, w)`.
        """
        if (self.nominal_features is None) | (self.background_masks is None):
            raise ValueError(f"Defect detector must be calibrated with a nominal sample.")
        if idx >= len(self.nominal_features):
            raise ValueError(
                f"Segment index can't be larger than number of segments, got {idx} and {len(self.nominal_features)}."
            )

        tgt_features = self.extract_features(x)
        ref_features = self.nominal_features[idx]

        amap = self.get_anomaly_map(tgt_features, ref_features, x.shape[-2:])
        amap = self.blur(amap)
        amap[:, :, self.background_masks[idx]] = 0
        return amap

    @torch.no_grad()
    def background_subtraction(self, x: torch.Tensor, background: torch.Tensor, thresh: int = 70) -> torch.Tensor:
        """Get foreground mask of given segment."""
        background_features = self.extract_features(background)
        tgt_features = self.extract_features(x)
        fg_map = self.get_anomaly_map(tgt_features, background_features, x.shape[-2:])

        fg_map = (255 * fg_map).squeeze().detach().cpu().numpy().astype(np.uint8)
        mask = cv2.threshold(fg_map, thresh, 255, cv2.THRESH_BINARY)[1]
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)

        fg_mask = np.zeros_like(fg_map)
        cv2.drawContours(fg_mask, [hull], 0, 255, -1)
        return torch.from_numpy(fg_mask > 0).to(background.device)
